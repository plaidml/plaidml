#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <iostream>

#include "openmp/runtime/src/kmp.h"

#define MKLOC(loc, routine)                                                    \
  static ident_t loc = {0, KMP_IDENT_KMPC, 0, 0, ";unknown;unknown;0;0;;"};

static void __kmp_GOMP_microtask_wrapper(int *gtid, int *npr,
                                         void (*task)(void *), void *data) {
  task(data);
}

static void __kmp_GOMP_fork_call(ident_t *loc, int gtid, unsigned num_threads,
                                 unsigned flags, void (*unwrapped_task)(void *),
                                 microtask_t wrapper, int argc, ...) {
  kmp_info_t *thr = __kmp_threads[gtid];
  kmp_team_t *team = thr->th.th_team;
  int tid = __kmp_tid_from_gtid(gtid);

  va_list ap;
  va_start(ap, argc);

  if (num_threads)
    __kmp_push_num_threads(loc, gtid, num_threads);
  if (flags)
    __kmp_push_proc_bind(loc, gtid, (kmp_proc_bind_t)flags);
  int rc = __kmp_fork_call(loc, gtid, fork_context_gnu, argc, wrapper,
                           __kmp_invoke_task_func, kmp_va_addr_of(ap));

  va_end(ap);

  if (rc) {
    __kmp_run_before_invoked_task(gtid, tid, thr, team);
  }
}

static void GOMP_parallel_start(void (*task)(void *), void *data,
                                unsigned num_threads) {
  int gtid = __kmp_entry_gtid();

  MKLOC(loc, "GOMP_parallel_start");
  KA_TRACE(20, ("GOMP_parallel_start: T#%d\n", gtid));
  __kmp_GOMP_fork_call(&loc, gtid, num_threads, 0u, task,
                       (microtask_t)__kmp_GOMP_microtask_wrapper, 2, task,
                       data);
}

static void GOMP_parallel_end(void) {
  int gtid = __kmp_get_gtid();
  kmp_info_t *thr = __kmp_threads[gtid];

  MKLOC(loc, "GOMP_parallel_end");
  KA_TRACE(20, ("GOMP_parallel_end: T#%d\n", gtid));

  if (!thr->th.th_team->t.t_serialized) {
    __kmp_run_after_invoked_task(gtid, __kmp_tid_from_gtid(gtid), thr,
                                 thr->th.th_team);
  }
  __kmp_join_call(&loc, gtid);
}

static void subfunction(void *data) {
  auto thread_num = __kmp_tid_from_gtid(__kmp_get_gtid());
  std::cout << "Thread id " << thread_num << " reporting in." << std::endl;
}

TEST(OpenMP, FourThreads) {
  int data = 0;
  unsigned int num_threads = 4;
  GOMP_parallel_start(subfunction, &data, num_threads);
  subfunction(&data);
  GOMP_parallel_end();
}
