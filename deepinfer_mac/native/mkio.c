// mkio: native two-priority chunked pread scheduler for expert streaming (no Python per chunk).
//
// A request reads `total` bytes at file offset `off` of `fd` into a scatter list of buffers. It is
// split into chunks of `chunk` bytes (cut points aligned to 16 KB). Worker threads always take
// demand ("high") chunks before speculative ("low") ones, and run at most `spec_conc` low chunks at
// once, so a demand read never queues behind a backlog of guesses. A low request can be promoted
// (its pending chunks jump the queue) or cancelled while none of its chunks has started.
//
// build: clang -O2 -shared -fPIC -o libmkio.dylib mkio.c
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/uio.h>
#include <unistd.h>

#define MAXREQ 4096
#define MAXIOV 16
#define ALIGNB 16384

typedef struct {
  int used;
  int fd;
  int nbuf;
  char *buf[MAXIOV];
  size_t len[MAXIOV];
  long long off;
  size_t total;
  int high;
  int nchunks, left, started, error, cancelled, done;
  long gen;
} Req;

typedef struct Job {
  int req;
  long gen;
  size_t a, b;  // byte range within the request
  struct Job *next, *prev;
} Job;

typedef struct {
  Job *head, *tail;
} Queue;

static Req reqs[MAXREQ];
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cv_work = PTHREAD_COND_INITIALIZER;
static pthread_cond_t cv_done = PTHREAD_COND_INITIALIZER;
static Queue qhi, qlo;
static int active_lo = 0, spec_conc = 2;
static size_t chunk_sz = 2u << 20;
static int next_slot = 0;
static long gen_ctr = 1;

static void q_push(Queue *q, Job *j) {
  j->next = NULL;
  j->prev = q->tail;
  if (q->tail) q->tail->next = j; else q->head = j;
  q->tail = j;
}
static void q_push_front(Queue *q, Job *j) {
  j->prev = NULL;
  j->next = q->head;
  if (q->head) q->head->prev = j; else q->tail = j;
  q->head = j;
}
static void q_unlink(Queue *q, Job *j) {
  if (j->prev) j->prev->next = j->next; else q->head = j->next;
  if (j->next) j->next->prev = j->prev; else q->tail = j->prev;
  j->next = j->prev = NULL;
}

static void do_read(Req *r, size_t a, size_t b, int *err) {
  // build the iovec slices covering [a, b) of the scatter list
  struct iovec iov[MAXIOV];
  int n = 0;
  size_t pos = 0;
  for (int i = 0; i < r->nbuf; i++) {
    size_t lo = a > pos ? a : pos, hi = b < pos + r->len[i] ? b : pos + r->len[i];
    if (lo < hi) {
      iov[n].iov_base = r->buf[i] + (lo - pos);
      iov[n].iov_len = hi - lo;
      n++;
    }
    pos += r->len[i];
  }
  off_t off = (off_t)(r->off + (long long)a);
  size_t want = b - a;
  int k = 0;
  while (want > 0 && k < n) {
    ssize_t got = preadv(r->fd, iov + k, n - k, off);
    if (got <= 0) {
      if (got < 0 && errno == EINTR) continue;
      *err = 1;
      return;
    }
    off += got;
    want -= (size_t)got;
    while (k < n && (size_t)got >= iov[k].iov_len) { got -= iov[k].iov_len; k++; }
    if (k < n && got > 0) { iov[k].iov_base = (char *)iov[k].iov_base + got; iov[k].iov_len -= got; }
  }
}

static void *worker(void *arg) {
  (void)arg;
  for (;;) {
    pthread_mutex_lock(&mu);
    Job *j = NULL;
    int lo = 0;
    for (;;) {
      if (qhi.head) { j = qhi.head; q_unlink(&qhi, j); break; }
      if (qlo.head && active_lo < spec_conc) { j = qlo.head; q_unlink(&qlo, j); lo = 1; active_lo++; break; }
      pthread_cond_wait(&cv_work, &mu);
    }
    Req *r = &reqs[j->req];
    r->started++;
    pthread_mutex_unlock(&mu);
    int err = 0;
    do_read(r, j->a, j->b, &err);
    pthread_mutex_lock(&mu);
    if (lo) { active_lo--; if (qlo.head) pthread_cond_signal(&cv_work); }
    if (err) r->error = 1;
    if (--r->left == 0) { r->done = 1; pthread_cond_broadcast(&cv_done); }
    pthread_mutex_unlock(&mu);
    free(j);
  }
  return NULL;
}

int mkio_init(int nthreads, long long chunk, int conc) {
  chunk_sz = (size_t)chunk;
  spec_conc = conc;
  for (int i = 0; i < nthreads; i++) {
    pthread_t t;
    if (pthread_create(&t, NULL, worker, NULL) != 0) return -1;
    pthread_detach(t);
  }
  return 0;
}

// returns request handle (slot index, >= 0) or -1
int mkio_submit(int fd, void **bufs, long long *lens, int nbuf, long long off, long long total, int high) {
  if (nbuf > MAXIOV) return -1;
  pthread_mutex_lock(&mu);
  int s = -1;
  for (int i = 0; i < MAXREQ; i++) {
    int c = (next_slot + i) % MAXREQ;
    if (!reqs[c].used) { s = c; break; }
  }
  if (s < 0) { pthread_mutex_unlock(&mu); return -1; }
  next_slot = (s + 1) % MAXREQ;
  Req *r = &reqs[s];
  memset(r, 0, sizeof(*r));
  r->used = 1;
  r->fd = fd;
  r->nbuf = nbuf;
  for (int i = 0; i < nbuf; i++) { r->buf[i] = (char *)bufs[i]; r->len[i] = (size_t)lens[i]; }
  r->off = off;
  r->total = (size_t)total;
  r->high = high;
  r->gen = gen_ctr++;
  size_t a = 0;
  int n = 0;
  while (a < r->total) {
    size_t b = a + chunk_sz;
    b = (b / ALIGNB) * ALIGNB;
    if (b <= a) b = a + ALIGNB;
    if (b > r->total) b = r->total;
    Job *j = (Job *)malloc(sizeof(Job));
    j->req = s; j->gen = r->gen; j->a = a; j->b = b;
    q_push(high ? &qhi : &qlo, j);
    n++;
    a = b;
  }
  r->nchunks = r->left = n;
  if (n == 0) r->done = 1;
  pthread_cond_broadcast(&cv_work);
  pthread_mutex_unlock(&mu);
  return s;
}

int mkio_done(int s) {
  pthread_mutex_lock(&mu);
  int d = reqs[s].done ? (reqs[s].error ? -1 : 1) : 0;
  pthread_mutex_unlock(&mu);
  return d;
}

// promote a low request: its pending chunks move to the front of the demand queue
void mkio_promote(int s) {
  pthread_mutex_lock(&mu);
  Req *r = &reqs[s];
  if (!r->high && !r->done) {
    r->high = 1;
    Job *moved[1024];
    int n = 0;
    for (Job *j = qlo.head; j && n < 1024;) {
      Job *nx = j->next;
      if (j->req == s && j->gen == r->gen) { q_unlink(&qlo, j); moved[n++] = j; }
      j = nx;
    }
    for (int i = n - 1; i >= 0; i--) q_push_front(&qhi, moved[i]);
    if (n) pthread_cond_broadcast(&cv_work);
  }
  pthread_mutex_unlock(&mu);
}

// wait for completion (call without the GIL); returns 1 ok, -1 error / cancelled
int mkio_wait(int s) {
  pthread_mutex_lock(&mu);
  Req *r = &reqs[s];
  if (!r->high && !r->done) {  // a needed guess: promote first
    pthread_mutex_unlock(&mu);
    mkio_promote(s);
    pthread_mutex_lock(&mu);
  }
  while (!r->done) pthread_cond_wait(&cv_done, &mu);
  int rc = r->error ? -1 : 1;
  pthread_mutex_unlock(&mu);
  return rc;
}

// cancel a low request none of whose chunks has started; 1 if cancelled
int mkio_cancel(int s) {
  pthread_mutex_lock(&mu);
  Req *r = &reqs[s];
  int ok = 0;
  if (!r->high && !r->done && r->started == 0) {
    for (Job *j = qlo.head; j;) {
      Job *nx = j->next;
      if (j->req == s && j->gen == r->gen) { q_unlink(&qlo, j); free(j); }
      j = nx;
    }
    r->done = 1;
    r->error = 1;
    r->cancelled = 1;
    r->left = 0;
    ok = 1;
    pthread_cond_broadcast(&cv_done);
  }
  pthread_mutex_unlock(&mu);
  return ok;
}

// release a finished request's slot
void mkio_release(int s) {
  pthread_mutex_lock(&mu);
  if (reqs[s].done) reqs[s].used = 0;
  pthread_mutex_unlock(&mu);
}
