// mkio: native two-priority chunked pread scheduler for expert streaming (no Python per chunk).
//
// A request reads `total` bytes at file offset `off` of `fd` into a scatter list of buffers. It is
// split into chunks of `chunk` bytes (cut points aligned to 16 KB). Worker threads always take
// demand ("high") chunks before speculative ("low") ones, and run at most `spec_conc` low chunks at
// once, so a demand read never queues behind a backlog of guesses. A low request can be promoted
// (its pending chunks jump the queue) or cancelled while none of its chunks has started.
//
// Optional tracing (off unless mkio_trace_enable is called): per chunk the worker thread, queue /
// start / end times and whether it ran at speculative priority; per request submit / promote / cancel
// events with a caller tag. Times are CLOCK_UPTIME_RAW ns, the clock behind Python's perf_counter_ns
// on macOS, so they line up with host-side events.
//
// build: clang -O2 -shared -fPIC -o libmkio.dylib mkio.c
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/uio.h>
#include <time.h>
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
  long long tag;
} Req;

typedef struct Job {
  int req;
  long gen;
  size_t a, b;  // byte range within the request
  long long tq;  // enqueue time (tracing)
  struct Job *next, *prev;
} Job;

// ---- tracing ----
typedef struct { long long tq, t0, t1, tag, a, b; long gen; int thr, lo; } TChunk;
typedef struct { long long t, tag, total; long gen; int kind, high; } TReq;  // kind 0 submit, 1 promote, 2 cancel
static TChunk *tch = NULL;
static TReq *trq = NULL;
static int tch_cap = 0, tch_n = 0, trq_cap = 0, trq_n = 0, trace_on = 0;

static long long now_ns(void) { return (long long)clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

static void trace_req(int kind, Req *r, long long t) {  // caller holds mu
  if (!trace_on || trq_n >= trq_cap) return;
  TReq *e = &trq[trq_n++];
  e->t = t; e->tag = r->tag; e->total = (long long)r->total; e->gen = r->gen; e->kind = kind; e->high = r->high;
}

typedef struct {
  Job *head, *tail;
} Queue;

static Req reqs[MAXREQ];
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cv_work = PTHREAD_COND_INITIALIZER;
static pthread_cond_t cv_done = PTHREAD_COND_INITIALIZER;
static Queue qhi, qlo;
static int active_lo = 0, active_hi = 0, spec_conc = 2;
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
  const int thr = (int)(intptr_t)arg;
  for (;;) {
    pthread_mutex_lock(&mu);
    Job *j = NULL;
    int lo = 0;
    for (;;) {
      if (qhi.head) { j = qhi.head; q_unlink(&qhi, j); active_hi++; break; }
      // guesses only while no demand chunk is queued or on the SSD: a demand read has the whole device
      if (qlo.head && active_hi == 0 && active_lo < spec_conc) { j = qlo.head; q_unlink(&qlo, j); lo = 1; active_lo++; break; }
      pthread_cond_wait(&cv_work, &mu);
    }
    Req *r = &reqs[j->req];
    r->started++;
    const int traced = trace_on;
    const long long t0 = traced ? now_ns() : 0;
    pthread_mutex_unlock(&mu);
    int err = 0;
    do_read(r, j->a, j->b, &err);
    const long long t1 = traced ? now_ns() : 0;
    pthread_mutex_lock(&mu);
    if (traced && trace_on && tch_n < tch_cap) {
      TChunk *e = &tch[tch_n++];
      e->tq = j->tq; e->t0 = t0; e->t1 = t1; e->tag = r->tag; e->a = (long long)j->a; e->b = (long long)j->b;
      e->gen = j->gen; e->thr = thr; e->lo = lo;
    }
    if (lo) { active_lo--; if (qlo.head) pthread_cond_signal(&cv_work); }
    else { active_hi--; if (active_hi == 0 && qlo.head) pthread_cond_broadcast(&cv_work); }
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
    if (pthread_create(&t, NULL, worker, (void *)(intptr_t)i) != 0) return -1;
    pthread_detach(t);
  }
  return 0;
}

// returns request handle (slot index, >= 0) or -1; `tag` is an opaque caller id kept for tracing
int mkio_submit_tagged(int fd, void **bufs, long long *lens, int nbuf, long long off, long long total, int high, long long tag) {
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
  r->tag = tag;
  const long long tq = trace_on ? now_ns() : 0;
  trace_req(0, r, tq);
  size_t a = 0;
  int n = 0;
  while (a < r->total) {
    size_t b = a + chunk_sz;
    b = (b / ALIGNB) * ALIGNB;
    if (b <= a) b = a + ALIGNB;
    if (b > r->total) b = r->total;
    Job *j = (Job *)malloc(sizeof(Job));
    j->req = s; j->gen = r->gen; j->a = a; j->b = b; j->tq = tq;
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

int mkio_submit(int fd, void **bufs, long long *lens, int nbuf, long long off, long long total, int high) {
  return mkio_submit_tagged(fd, bufs, lens, nbuf, off, total, high, -1);
}

// generation (unique id) of the request in slot s; valid until the slot is released
long mkio_gen(int s) {
  pthread_mutex_lock(&mu);
  long g = reqs[s].gen;
  pthread_mutex_unlock(&mu);
  return g;
}

long long mkio_now(void) { return now_ns(); }

// start (or restart) tracing with room for `cap` chunk and `cap` request events
int mkio_trace_enable(int cap) {
  pthread_mutex_lock(&mu);
  if (cap > tch_cap) {
    free(tch); free(trq);
    tch = (TChunk *)calloc((size_t)cap, sizeof(TChunk));
    trq = (TReq *)calloc((size_t)cap, sizeof(TReq));
    if (!tch || !trq) { free(tch); free(trq); tch = NULL; trq = NULL; tch_cap = trq_cap = 0; pthread_mutex_unlock(&mu); return -1; }
    tch_cap = trq_cap = cap;
  }
  tch_n = trq_n = 0;
  trace_on = 1;
  pthread_mutex_unlock(&mu);
  return 0;
}

void mkio_trace_disable(void) {
  pthread_mutex_lock(&mu);
  trace_on = 0;
  pthread_mutex_unlock(&mu);
}

// copy up to `max` chunk events as 9 int64 each: tq, t0, t1, gen, tag, thr, lo, a, b; returns the count
int mkio_trace_chunks(long long *out, int max) {
  pthread_mutex_lock(&mu);
  int n = tch_n < max ? tch_n : max;
  for (int i = 0; i < n; i++) {
    TChunk *e = &tch[i];
    long long *o = out + 9 * i;
    o[0] = e->tq; o[1] = e->t0; o[2] = e->t1; o[3] = e->gen; o[4] = e->tag; o[5] = e->thr; o[6] = e->lo; o[7] = e->a; o[8] = e->b;
  }
  pthread_mutex_unlock(&mu);
  return n;
}

// copy up to `max` request events as 6 int64 each: t, gen, tag, kind, high, total; returns the count
int mkio_trace_reqs(long long *out, int max) {
  pthread_mutex_lock(&mu);
  int n = trq_n < max ? trq_n : max;
  for (int i = 0; i < n; i++) {
    TReq *e = &trq[i];
    long long *o = out + 6 * i;
    o[0] = e->t; o[1] = e->gen; o[2] = e->tag; o[3] = e->kind; o[4] = e->high; o[5] = e->total;
  }
  pthread_mutex_unlock(&mu);
  return n;
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
    trace_req(1, r, trace_on ? now_ns() : 0);
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
    trace_req(2, r, trace_on ? now_ns() : 0);
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
