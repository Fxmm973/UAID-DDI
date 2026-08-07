#!/usr/bin/env Python
# coding=utf-8
"""
实验记录模块 - 用于记录实验过程的详细信息
"""

import os
import json
from datetime import datetime
from collections import defaultdict


class ExperimentRecorder:
    """实验记录器 - 记录实验过程中的所有关键信息"""
    
    def __init__(self, project_name="RareDDIE", result_file="result.txt"):
        """
        初始化实验记录器
        
        Args:
            project_name: 项目名称
            result_file: 结果保存文件名
        """
        self.project_name = project_name
        self.result_file = result_file
        self.start_time = datetime.now()
        self.experiment_data = {
            'project_name': project_name,
            'start_time': self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'hyperparameters': {},
            'training_history': [],
            'evaluation_results': {},
            'best_models': {},
            'test_results': {}
        }
        
    def record_hyperparameters(self, args):
        """记录超参数"""
        hyperparams = {}
        for k, v in vars(args).items():
            if k not in ['save_path']:  # 排除路径等不需要记录的内容
                hyperparams[k] = str(v)
        self.experiment_data['hyperparameters'] = hyperparams
        self._write_to_file()
        
    def record_training_step(self, batch_num, loss, metrics=None):
        """
        记录训练步骤
        
        Args:
            batch_num: 批次编号
            loss: 损失值
            metrics: 评估指标字典 {'acc', 'auroc', 'f1_score', ...}
        """
        step_data = {
            'batch_num': batch_num,
            'loss': float(loss),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        if metrics:
            step_data['metrics'] = {k: float(v) for k, v in metrics.items()}
        self.experiment_data['training_history'].append(step_data)
        
        # 每记录一定步数后写入文件（避免频繁IO）
        if len(self.experiment_data['training_history']) % 100 == 0:
            self._write_to_file()
    
    def record_evaluation(self, mode, metrics, is_best=False, batch_num=None):
        """
        记录评估结果
        
        Args:
            mode: 评估模式 ('dev', 'test', 'test2')
            metrics: 评估指标字典
            is_best: 是否为最佳模型
            batch_num: 批次编号
        """
        eval_data = {
            'mode': mode,
            'metrics': {k: float(v) for k, v in metrics.items()},
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        if batch_num is not None:
            eval_data['batch_num'] = batch_num
            
        if mode not in self.experiment_data['evaluation_results']:
            self.experiment_data['evaluation_results'][mode] = []
        self.experiment_data['evaluation_results'][mode].append(eval_data)
        
        if is_best:
            self.experiment_data['best_models'][mode] = {
                'batch_num': batch_num,
                'metrics': {k: float(v) for k, v in metrics.items()},
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        self._write_to_file()
    
    def record_test_result(self, test_name, metrics):
        """
        记录测试结果
        
        Args:
            test_name: 测试名称
            metrics: 评估指标字典
        """
        self.experiment_data['test_results'][test_name] = {
            'metrics': {k: float(v) for k, v in metrics.items()},
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self._write_to_file()
    
    def finalize(self):
        """完成实验记录"""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        self.experiment_data['end_time'] = self.end_time.strftime('%Y-%m-%d %H:%M:%S')
        self.experiment_data['duration_seconds'] = duration
        self.experiment_data['duration_formatted'] = self._format_duration(duration)
        
        self._write_to_file()
        return self._format_final_report()
    
    def _format_duration(self, seconds):
        """格式化时间长度"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def _write_to_file(self):
        """将实验数据写入文件"""
        report = self._format_report()
        with open(self.result_file, 'w', encoding='utf-8') as f:
            f.write(report)
    
    def _format_report(self):
        """格式化报告内容"""
        lines = []
        lines.append("=" * 80)
        lines.append(f"实验记录 - {self.project_name}")
        lines.append("=" * 80)
        lines.append("")
        
        # 基本信息
        lines.append("【实验基本信息】")
        lines.append(f"项目名称: {self.experiment_data['project_name']}")
        lines.append(f"开始时间: {self.experiment_data['start_time']}")
        if 'end_time' in self.experiment_data:
            lines.append(f"结束时间: {self.experiment_data['end_time']}")
            lines.append(f"实验时长: {self.experiment_data['duration_formatted']}")
        lines.append("")
        
        # 超参数
        lines.append("【超参数配置】")
        for k, v in sorted(self.experiment_data['hyperparameters'].items()):
            lines.append(f"  {k}: {v}")
        lines.append("")
        
        # 最佳模型
        if self.experiment_data['best_models']:
            lines.append("【最佳模型记录】")
            for mode, best_info in self.experiment_data['best_models'].items():
                lines.append(f"  {mode.upper()} 最佳模型 (Batch {best_info.get('batch_num', 'N/A')}):")
                for metric, value in best_info['metrics'].items():
                    lines.append(f"    {metric}: {value:.4f}")
                lines.append(f"    时间: {best_info['timestamp']}")
                lines.append("")
        
        # 评估结果摘要
        if self.experiment_data['evaluation_results']:
            lines.append("【评估结果摘要】")
            for mode, eval_list in self.experiment_data['evaluation_results'].items():
                if eval_list:
                    latest = eval_list[-1]
                    lines.append(f"  {mode.upper()} 最新评估结果:")
                    for metric, value in latest['metrics'].items():
                        lines.append(f"    {metric}: {value:.4f}")
                    lines.append("")
        
        # 测试结果
        if self.experiment_data['test_results']:
            lines.append("【测试结果】")
            for test_name, test_info in self.experiment_data['test_results'].items():
                lines.append(f"  {test_name}:")
                for metric, value in test_info['metrics'].items():
                    lines.append(f"    {metric}: {value:.4f}")
                lines.append("")
        
        # 训练历史（最近10条）
        if self.experiment_data['training_history']:
            lines.append("【训练历史 (最近10条)】")
            recent_history = self.experiment_data['training_history'][-10:]
            for step in recent_history:
                lines.append(f"  Batch {step['batch_num']}: Loss={step['loss']:.4f}")
                if 'metrics' in step:
                    metric_str = ", ".join([f"{k}={v:.4f}" for k, v in step['metrics'].items()])
                    lines.append(f"    Metrics: {metric_str}")
            lines.append(f"  ... (共 {len(self.experiment_data['training_history'])} 条记录)")
            lines.append("")
        
        lines.append("=" * 80)
        return "\n".join(lines)
    
    def _format_final_report(self):
        """格式化最终报告"""
        return self._format_report()




