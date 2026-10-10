/* =========================================================================
 * ds4_distributed.c - Distributed inference runtime.
 * =========================================================================
 *
 * This module owns the DS4 distributed transport and orchestration layer. The
 * rest of the engine still sees a normal ds4_session: when distributed mode is
 * active, ds4.c delegates sync/eval/save/load to the coordinator session API in
 * this file.
 *
 * Workers execute contiguous model slices with the same graph-slice entry
 * points used by the local engine. KV snapshots remain topology-independent:
 * save gathers worker-owned layer tensors into the normal DSV4 payload, and
 * load splits a normal DSV4 payload across the currently registered route.
 */

#include "ds4_distributed.h"

#include <arpa/inet.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <netdb.h>
#include <net/if.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
/* ds4.c and ds4_server.c still call these behind a runtime `s->distributed`
 * check that is never true in this build. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
int ds4_dist_session_create(
        ds4_dist_session **out,
        ds4_engine *engine,
        const ds4_dist_options *opt,
        ds4_session *owner,
        int ctx_size,
        char *err,
        size_t errlen) {
    if (err && errlen) snprintf(err, errlen, "distributed execution is not part of this build");
    return -1;
}
int ds4_dist_session_eval(
        ds4_dist_session *d,
        ds4_session *owner,
        const ds4_tokens *checkpoint,
        int token,
        float *logits,
        char *err,
        size_t errlen) {
    if (err && errlen) snprintf(err, errlen, "distributed execution is not part of this build");
    return -1;
}
void ds4_dist_session_free(ds4_dist_session *d) { (void)d; }
int ds4_dist_session_load_payload(
        ds4_dist_session *d,
        ds4_session *owner,
        FILE *fp,
        uint64_t payload_bytes,
        char *err,
        size_t errlen) {
    if (err && errlen) snprintf(err, errlen, "distributed execution is not part of this build");
    return -1;
}
int ds4_dist_session_save_payload(
        ds4_dist_session *d,
        ds4_session *owner,
        FILE *fp,
        char *err,
        size_t errlen) {
    if (err && errlen) snprintf(err, errlen, "distributed execution is not part of this build");
    return -1;
}
int ds4_dist_session_sync(
        ds4_dist_session *d,
        ds4_session *owner,
        const ds4_tokens *checkpoint,
        const ds4_tokens *prompt,
        float *logits,
        char *err,
        size_t errlen) {
    if (err && errlen) snprintf(err, errlen, "distributed execution is not part of this build");
    return -1;
}
bool ds4_dist_enabled(const ds4_dist_options *opt) { return false; }
ds4_dist_options *ds4_dist_options_create(void) { return calloc(1, sizeof(ds4_dist_options)); }
void ds4_dist_options_free(ds4_dist_options *opt) { free(opt); }
void ds4_dist_usage(FILE *fp) { }
ds4_dist_cli_parse_result ds4_dist_parse_cli_arg(const char *arg, int *index, int argc, char **argv,
        ds4_dist_options *opt, char *err, size_t errlen) { return DS4_DIST_CLI_NOT_MATCHED; }
int ds4_dist_prepare_engine_options(const ds4_dist_options *opt, ds4_engine_options *engine,
        char *err, size_t errlen) { return 0; }
int ds4_dist_session_route_ready(ds4_dist_session *d, char *err, size_t errlen) { return -1; }
int ds4_dist_run(ds4_engine *engine, const ds4_dist_options *opt, const ds4_dist_generation_options *gen) {
    fprintf(stderr, "ds4: distributed execution is not part of this build\n");
    return 1;
}
#pragma clang diagnostic pop
