"""Single-client WSM PoC. Deploy only under a bounded owner that stops its apps."""

from spindle.configs.qwen3_4b_lora_32k import Config as Parent


class Config(Parent):
    name = "wsm-qwen3-4b-lora-16k"
    model = "Qwen/Qwen3-4B"
    max_context_length = 16384
    trainer_gpu = "H100"
    trainer_gpus_per_node = 1
    trainer_nodes = 1
    trainer_cpu = 8
    trainer_memory_mib = 32768
    trainer_max_instances = 1
    trainer_max_clients_per_instance = 1
    trainer_max_clients_per_batch = 1
    trainer_timeout_s = 900
    inference_gpu = "H100"
    inference_gpus_per_node = 1
    inference_cpu = 8
    inference_memory_mib = 32768
    inference_min_replicas = 0
    inference_max_replicas = 1
    inference_target_concurrency = 1
    inference_scaledown_window_s = 1
    inference_startup_timeout_s = 600
    session_idle_timeout_s = 120
    pool_idle_timeout_s = 30
    sweep_interval_s = 15
    overrides = {
        "miles_cfg.model_type": "qwen3-4B",
        "miles_cfg.tensor_model_parallel_size": 1,
        "miles_cfg.max_lora_slots": 1,
        "miles_cfg.max_lora_rank": 8,
        "miles_cfg.default_lora_alpha": 8,
        "miles_cfg.max_tokens_per_gpu": 16384,
        "sglang_cfg.tp_size": 1,
        "sglang_cfg.max_running_requests": 1,
        "sglang_cfg.max_queued_requests": 2,
        "sglang_cfg.max_loaded_loras": 1,
        "sglang_cfg.max_loras_per_batch": 1,
        "platform.frontend": "wsm-spindle-poc",
        "platform.modal.environment": "main",
        "platform.secrets.api": "wsm-spindle-poc-api",
        "platform.secrets.sampler_proxy": "wsm-spindle-poc-proxy",
        "platform.storage.assets": "wsm-spindle-poc-assets",
        "platform.storage.checkpoints": "wsm-spindle-poc-checkpoints",
        "platform.storage.bulletin": "wsm-spindle-poc-bulletin",
    }


config = Config()
