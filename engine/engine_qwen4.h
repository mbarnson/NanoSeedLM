// engine/engine_qwen4.h - included before engine_api.h: its names become the qwen4 adapter's (engine/qwen4_api.c), so
// one binary can hold it beside the MoVA engine.
#define Eng Q4Api
#define eng_open q4_eng_open
#define eng_close q4_eng_close
#define eng_describe q4_eng_describe
#define eng_vocab q4_eng_vocab
#define eng_mem q4_eng_mem
#define eng_prefill q4_eng_prefill
#define eng_prefill_cached q4_eng_prefill_cached
#define eng_prefill_begin q4_eng_prefill_begin
#define eng_prefill_next q4_eng_prefill_next
#define eng_rewind q4_eng_rewind
#define eng_fork q4_eng_fork
#define eng_free q4_eng_free
#define eng_len q4_eng_len
#define eng_generate q4_eng_generate
#define eng_generate_pick q4_eng_generate_pick
#define eng_step q4_eng_step
#define eng_push q4_eng_push
#define eng_step_batch q4_eng_step_batch
#define eng_kv_bytes q4_eng_kv_bytes
#define eng_kv_read q4_eng_kv_read
#define eng_kv_write q4_eng_kv_write
#define eng_score q4_eng_score
