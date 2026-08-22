import json
import os


def generate_common_test_set():
    base_path = os.path.dirname(os.path.abspath(__file__))

    test_file = os.path.join(base_path, 'test_tasks.json')

    output_common = os.path.join(base_path, 'common_test_tasks.json')
    output_uncommon = os.path.join(base_path, 'uncommon_test_tasks.json')

    if not os.path.exists(test_file):
        print(f"❌ 错误: 找不到测试集文件 {test_file}")
        return

    print(f"正在读取测试集: {test_file} ...")
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    rel_counts = {rel: len(samples) for rel, samples in test_data.items()}

    threshold = 50

    print("\n测试集关系样本分布情况 (Top 5):")
    sorted_counts = sorted(rel_counts.items(), key=lambda x: x[1], reverse=True)
    for rel, count in sorted_counts[:5]:
        print(f" - {rel[:50]}: {count} 个样本")

    common_test_data = {}
    uncommon_test_data = {}

    for rel, samples in test_data.items():
        if len(samples) > threshold:
            common_test_data[rel] = samples
        else:
            uncommon_test_data[rel] = samples

    if not common_test_data:
        print(f"\n⚠️ 警告: 测试集中没有样本量超过 {threshold} 的关系。")
        print("💡 正在执行自动补救：提取测试集中样本量最大的前 3 个关系作为常见类...")
        for rel, count in sorted_counts[:3]:
            common_test_data[rel] = test_data[rel]
            if rel in uncommon_test_data:
                del uncommon_test_data[rel]

    print("\n" + "=" * 50)
    with open(output_common, 'w', encoding='utf-8') as f:
        json.dump(common_test_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 常见测试集已生成: {output_common} (包含 {len(common_test_data)} 个关系)")

    with open(output_uncommon, 'w', encoding='utf-8') as f:
        json.dump(uncommon_test_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 非常见测试集已生成: {output_uncommon} (包含 {len(uncommon_test_data)} 个关系)")
    print("=" * 50)

    print("\n🚀 处理完成！现在你可以去运行训练程序了。")


if __name__ == "__main__":
    generate_common_test_set()