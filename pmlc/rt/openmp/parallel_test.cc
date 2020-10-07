// RUN: parallel_test | FileCheck %s

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <iostream>

#include "openmp/runtime/src/kmp.h"
#include "openmp/runtime/src/kmp_atomic.h"

#if OMPT_SUPPORT
#include "openmp/runtime/src/ompt-specific.h"
#endif

#include "plaidml/testenv.h"

class ParallelTest : public ::plaidml::edsl::TestFixture {};

extern "C" {

#define MKLOC(loc, routine)                                                    \
  static ident_t loc = {0, KMP_IDENT_KMPC, 0, 0, ";unknown;unknown;0;0;;"};

void __kmp_GOMP_microtask_wrapper(int *gtid, int *npr, void (*task)(void *),
                                  void *data) {
#if OMPT_SUPPORT
  kmp_info_t *thr;
  ompt_frame_t *ompt_frame;
  ompt_state_t enclosing_state;

  if (ompt_enabled.enabled) {
    // get pointer to thread data structure
    thr = __kmp_threads[*gtid];

    // save enclosing task state; set current state for task
    enclosing_state = thr->th.ompt_thread_info.state;
    thr->th.ompt_thread_info.state = ompt_state_work_parallel;

    // set task frame
    __ompt_get_task_info_internal(0, NULL, NULL, &ompt_frame, NULL, NULL);
    ompt_frame->exit_frame.ptr = OMPT_GET_FRAME_ADDRESS(0);
  }
#endif

  task(data);

#if OMPT_SUPPORT
  if (ompt_enabled.enabled) {
    // clear task frame
    ompt_frame->exit_frame = ompt_data_none;

    // restore enclosing state
    thr->th.ompt_thread_info.state = enclosing_state;
  }
#endif
}

static void __kmp_GOMP_fork_call(ident_t *loc, int gtid, unsigned num_threads,
                                 unsigned flags, void (*unwrapped_task)(void *),
                                 microtask_t wrapper, int argc, ...) {
  int rc;
  kmp_info_t *thr = __kmp_threads[gtid];
  kmp_team_t *team = thr->th.th_team;
  int tid = __kmp_tid_from_gtid(gtid);

  va_list ap;
  va_start(ap, argc);

  if (num_threads != 0)
    __kmp_push_num_threads(loc, gtid, num_threads);
  if (flags != 0)
    __kmp_push_proc_bind(loc, gtid, (kmp_proc_bind_t)flags);
  rc = __kmp_fork_call(loc, gtid, fork_context_gnu, argc, wrapper,
                       __kmp_invoke_task_func, kmp_va_addr_of(ap));

  va_end(ap);

  if (rc) {
    __kmp_run_before_invoked_task(gtid, tid, thr, team);
  }

#if OMPT_SUPPORT
  int ompt_team_size;
  if (ompt_enabled.enabled) {
    ompt_team_info_t *team_info = __ompt_get_teaminfo(0, NULL);
    ompt_task_info_t *task_info = __ompt_get_task_info_object(0);

    // implicit task callback
    if (ompt_enabled.ompt_callback_implicit_task) {
      ompt_team_size = __kmp_team_from_gtid(gtid)->t.t_nproc;
      ompt_callbacks.ompt_callback(ompt_callback_implicit_task)(
          ompt_scope_begin, &(team_info->parallel_data),
          &(task_info->task_data), ompt_team_size, __kmp_tid_from_gtid(gtid),
          ompt_task_implicit); // TODO: Can this be ompt_task_initial?
      task_info->thread_num = __kmp_tid_from_gtid(gtid);
    }
    thr->th.ompt_thread_info.state = ompt_state_work_parallel;
  }
#endif
}

void GOMP_parallel_start(void (*task)(void *), void *data,
                         unsigned num_threads) {
  int gtid = __kmp_entry_gtid();

#if OMPT_SUPPORT
  ompt_frame_t *parent_frame, *frame;

  if (ompt_enabled.enabled) {
    __ompt_get_task_info_internal(0, NULL, NULL, &parent_frame, NULL, NULL);
    parent_frame->enter_frame.ptr = OMPT_GET_FRAME_ADDRESS(0);
    OMPT_STORE_RETURN_ADDRESS(gtid);
  }
#endif

  MKLOC(loc, "GOMP_parallel_start");
  KA_TRACE(20, ("GOMP_parallel_start: T#%d\n", gtid));
  __kmp_GOMP_fork_call(&loc, gtid, num_threads, 0u, task,
                       (microtask_t)__kmp_GOMP_microtask_wrapper, 2, task,
                       data);
#if OMPT_SUPPORT
  if (ompt_enabled.enabled) {
    __ompt_get_task_info_internal(0, NULL, NULL, &frame, NULL, NULL);
    frame->exit_frame.ptr = OMPT_GET_FRAME_ADDRESS(0);
  }
#endif
}

void GOMP_parallel_end(void) {
  int gtid = __kmp_get_gtid();
  kmp_info_t *thr;

  thr = __kmp_threads[gtid];

  MKLOC(loc, "GOMP_parallel_end");
  KA_TRACE(20, ("GOMP_parallel_end: T#%d\n", gtid));

  if (!thr->th.th_team->t.t_serialized) {
    __kmp_run_after_invoked_task(gtid, __kmp_tid_from_gtid(gtid), thr,
                                 thr->th.th_team);
  }
#if OMPT_SUPPORT
  if (ompt_enabled.enabled) {
    // Implicit task is finished here, in the barrier we might schedule
    // deferred tasks,
    // these don't see the implicit task on the stack
    OMPT_CUR_TASK_INFO(thr)->frame.exit_frame = ompt_data_none;
  }
#endif

  __kmp_join_call(&loc, gtid
#if OMPT_SUPPORT
                  ,
                  fork_context_gnu
#endif
  ); // NOLINT
}

} // extern "C"

void subfunction(void *data) {
  auto thread_num = __kmp_tid_from_gtid(__kmp_get_gtid());
  std::cout << "Thread id " << thread_num << " reporting in." << std::endl;
}

TEST_F(ParallelTest, FourThreads) {
  int data = 0;
  unsigned int num_threads = 4;
  GOMP_parallel_start(subfunction, &data, num_threads);
  // CHECK: Thread id [0-3] reporting in
  subfunction(&data);
  GOMP_parallel_end();
}
