import argparse

def read_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset1", type=str)
    parser.add_argument("--embed_dim", default=128, type=int) #CSE 原子层输出维度（≈ 图 1a 的 d）。
    parser.add_argument("--few", default=10, type=int)#	测试阶段每个事件的 support-set 样本数（K-shot 的 K）。图 3 实验里 1-shot/5-shot 就是靠改它。
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--neg_num", default=1, type=int)
    parser.add_argument("--random_embed", default=False, type=bool)#	True=不用预训练 TransE，全部随机初始化；False=先 TransE 预训练 200 epoch。
    parser.add_argument("--train_few", default=10, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--max_batches", default=40000, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--log_every", default=50, type=int)#日志，	每 50 个 batch 打印一次 loss
    parser.add_argument("--eval_every", default=1000, type=int) #每 1 000 batch 在验证集测一次 AUC，用来早停
    parser.add_argument("--fine_tune", default=True, type=bool)#迁移学习 是否“先 DDIE 预训练 → 再 synergy 微调”；图 4 的 10-shot w/ transfer 即靠它。

    parser.add_argument("--aggregate", default='max', type=str)
    parser.add_argument("--max_neighbor", default=30, type=int) #每个药最多取 30 个邻居进图，再大显存炸。罕见事件可降到 15。
    parser.add_argument("--no_meta", action='store_true')#消融
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--grad_clip", default=5.0, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--embed_model", default='TransE', type=str) # ComplEx 知识图谱预训练方法；论文默认 TransE
    parser.add_argument("--prefix", default='intial', type=str)
    parser.add_argument("--seed", default='19940419', type=int)

    args = parser.parse_args()
    args.save_path = 'models/' + args.prefix

    print("------HYPERPARAMETERS-------")
    for k, v in vars(args).items():
        print(k + ': ' + str(v))
    print("----------------------------")

    return args

