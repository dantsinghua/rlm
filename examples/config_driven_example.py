"""
RLM 配置驱动模式示例

演示如何使用 config.yaml 配置文件驱动的 RLM 初始化。
支持多模型、角色映射、迭代多样性等功能。
"""

import os
from dotenv import load_dotenv

from rlm import RLM, load_config
from rlm.config.loader import DEFAULT_CONFIG_PATH
from rlm.logger import RLMLogger

load_dotenv()


def demo_config_loading():
    """演示配置加载功能"""
    print("=" * 60)
    print("配置加载演示")
    print("=" * 60)

    # 加载默认配置
    config = load_config()

    print(f"\n默认配置文件: {DEFAULT_CONFIG_PATH}")
    print(f"\n已配置的提供商: {list(config.providers.keys())}")
    print(f"已配置的模型: {list(config.models.keys())}")
    print(f"已配置的角色: {list(config.roles.keys())}")

    # 查看角色映射
    print("\n角色映射:")
    for role in ["completion_turn", "sub_query", "default_answer", "fallback_answer"]:
        model = config.get_model_for_role(role)
        fallback = config.get_fallback_for_role(role)
        print(f"  {role}: {model}" + (f" (fallback: {fallback})" if fallback else ""))

    # 查看豆包模型配置
    print("\n豆包模型配置:")
    doubao = config.get_model("doubao-seed-1.6")
    provider = config.get_provider(doubao.provider)
    print(f"  模型名: {doubao.model_name}")
    print(f"  提供商: {doubao.provider}")
    print(f"  base_url: {provider.base_url}")
    print(f"  参数: {doubao.params}")
    print(f"  extra_body: {doubao.extra_body}")


def demo_rlm_initialization():
    """演示 RLM 配置驱动初始化"""
    print("\n" + "=" * 60)
    print("RLM 配置驱动初始化演示")
    print("=" * 60)

    # 方式1: 使用配置对象
    config = load_config()
    logger = RLMLogger(log_dir="./logs")

    rlm = RLM(
        config=config,
        environment="local",
        max_iterations=10,
        logger=logger,
        verbose=True,
    )

    print(f"\n配置模式: {rlm._config_mode}")
    print(f"最大迭代次数: {rlm.max_iterations}")
    print(f"环境类型: {rlm.environment_type}")

    # 演示迭代模型选择
    print("\n迭代模型选择 (多样性关闭):")
    for i in range(3):
        model = rlm._get_model_for_iteration(i)
        print(f"  迭代 {i}: {model}")

    # 启用迭代多样性
    rlm._config.iteration_diversity.enabled = True
    print("\n迭代模型选择 (多样性开启 - round_robin):")
    for i in range(6):
        model = rlm._get_model_for_iteration(i)
        print(f"  迭代 {i}: {model}")


def demo_completion_with_config():
    """演示使用配置驱动模式进行补全"""
    print("\n" + "=" * 60)
    print("RLM 补全调用演示")
    print("=" * 60)

    # 检查是否有 API 密钥
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ARK_API_KEY")
    if not api_key:
        print("\n[警告] 未找到 API 密钥，跳过实际调用演示")
        print("请设置 OPENAI_API_KEY 或 ARK_API_KEY 环境变量")
        return

    config = load_config()
    rlm = RLM(
        config=config,
        environment="local",
        max_iterations=5,
        verbose=True,
    )

    # 简单的补全调用
    result = rlm.completion(
        prompt="What is 2 + 2?",
        root_prompt="Answer the math question.",
    )

    print(f"\n结果: {result.response}")
    print(f"执行时间: {result.execution_time:.2f}s")


if __name__ == "__main__":
    demo_config_loading()
    demo_rlm_initialization()
    demo_completion_with_config()
