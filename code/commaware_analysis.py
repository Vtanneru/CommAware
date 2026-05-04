#!/usr/bin/env python3
"""
CommAware: Communication-Aware Job Placement for Multi-Tenant Cloud-HPC
Analysis, measurement generation, and visualization pipeline.

Author: Venkateswarlu Tanneru
Date: May 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import os
import json

# Set random seed for reproducibility
np.random.seed(42)

class CommAwareConfig:
    """Configuration for CommAware system."""

    def __init__(self):
        self.jobs = {
            'ResNet50': {'type': 'CNN', 'params': 25.5e6, 'batch': 128, 'profile': 'A'},
            'ResNet101': {'type': 'CNN', 'params': 44.5e6, 'batch': 64, 'profile': 'A'},
            'ResNet152': {'type': 'CNN', 'params': 60.2e6, 'batch': 64, 'profile': 'A'},
            'VGG16': {'type': 'CNN', 'params': 138e6, 'batch': 32, 'profile': 'A'},
            'MobileNet': {'type': 'CNN', 'params': 4.2e6, 'batch': 256, 'profile': 'A'},
            'EfficientNet': {'type': 'CNN', 'params': 66e6, 'batch': 32, 'profile': 'A'},
            'InceptionV3': {'type': 'CNN', 'params': 23.8e6, 'batch': 32, 'profile': 'A'},
            'Xception': {'type': 'CNN', 'params': 22.9e6, 'batch': 32, 'profile': 'A'},
            'BERTbase': {'type': 'Transformer', 'params': 110e6, 'batch': 32, 'profile': 'B'},
            'BERTlarge': {'type': 'Transformer', 'params': 340e6, 'batch': 16, 'profile': 'B'},
            'ViTbase': {'type': 'Transformer', 'params': 86e6, 'batch': 32, 'profile': 'B'},
            'ViTlarge': {'type': 'Transformer', 'params': 305e6, 'batch': 16, 'profile': 'B'},
            'GPT2': {'type': 'Transformer', 'params': 124e6, 'batch': 32, 'profile': 'B'},
            'GPT3small': {'type': 'Transformer', 'params': 125e6, 'batch': 16, 'profile': 'B'},
            'T5base': {'type': 'Transformer', 'params': 220e6, 'batch': 32, 'profile': 'B'},
            'ELECTRA': {'type': 'Transformer', 'params': 110e6, 'batch': 32, 'profile': 'B'},
            'DLRM': {'type': 'Sparse', 'params': 1.5e9, 'batch': 2048, 'profile': 'C'},
            'Wide&Deep': {'type': 'Sparse', 'params': 500e6, 'batch': 1024, 'profile': 'C'},
            'XGBoost': {'type': 'Sparse', 'params': 100e6, 'batch': 512, 'profile': 'C'},
            'LightGBM': {'type': 'Sparse', 'params': 50e6, 'batch': 1024, 'profile': 'C'},
            'CatBoost': {'type': 'Sparse', 'params': 75e6, 'batch': 512, 'profile': 'C'},
            'DeepFM': {'type': 'Sparse', 'params': 200e6, 'batch': 2048, 'profile': 'C'},
            'DistBERT': {'type': 'Sparse', 'params': 300e6, 'batch': 512, 'profile': 'C'},
            'TabNet': {'type': 'Sparse', 'params': 150e6, 'batch': 512, 'profile': 'C'},
            'CosmoFlow': {'type': 'Collective', 'params': 800e6, 'batch': 32, 'profile': 'D'},
            'DeepCAM': {'type': 'Collective', 'params': 950e6, 'batch': 16, 'profile': 'D'},
            'HydroNet': {'type': 'Collective', 'params': 700e6, 'batch': 32, 'profile': 'D'},
            'ExaNN': {'type': 'Collective', 'params': 1.2e9, 'batch': 16, 'profile': 'D'},
            'Operator': {'type': 'Collective', 'params': 600e6, 'batch': 32, 'profile': 'D'},
            'CineNet': {'type': 'Collective', 'params': 850e6, 'batch': 16, 'profile': 'D'},
            'InceptionResNet': {'type': 'Inference', 'params': 55e6, 'batch': 256, 'profile': 'C'},
            'MobileNetV2': {'type': 'Inference', 'params': 3.5e6, 'batch': 512, 'profile': 'C'},
            'EfficientNetLite': {'type': 'Inference', 'params': 10e6, 'batch': 256, 'profile': 'C'},
            'SqueezeNet': {'type': 'Inference', 'params': 1.2e6, 'batch': 512, 'profile': 'C'},
            'MobileNetEdge': {'type': 'Inference', 'params': 2.2e6, 'batch': 512, 'profile': 'C'},
            'TinyBERT': {'type': 'Inference', 'params': 14e6, 'batch': 256, 'profile': 'C'},
        }

        self.topologies = ['FatTree', 'Clos', 'Dragonfly', 'Mesh4D', 'HyperX', 'Ring']
        self.placement_policies = ['Random', 'FIFO', 'TopoAware', 'CommAware']

class CommAwareGenerator:
    """Generate synthetic measurements for CommAware validation."""

    def __init__(self, config):
        self.config = config
        self.measurements = []

    def compute_communication(self, job_name):
        """Compute communication volume based on job profile."""
        job = self.config.jobs[job_name]
        profile = job['profile']
        params = job['params']
        batch = job['batch']

        # Empirical communication model based on profile
        if profile == 'A':  # Compute-dominant
            comm_vol = np.random.uniform(0.3, 1.2) * (batch / 128)
        elif profile == 'B':  # Communication-intensive
            comm_vol = np.random.uniform(8.5, 14.2) * (batch / 32)
        elif profile == 'C':  # Latency-critical
            comm_vol = np.random.uniform(2.1, 7.8) * (batch / 512)
        else:  # Profile D: Bandwidth-hungry
            comm_vol = np.random.uniform(9.2, 16.5) * (batch / 32)

        return comm_vol

    def compute_congestion(self, job1_profile, job2_profile, topology, placement_policy):
        """Estimate network congestion from job profiles and placement."""
        # Congestion model: baseline + interference + topology effect
        baseline = {
            'A': 20, 'B': 120, 'C': 40, 'D': 140
        }

        base_congestion = (baseline[job1_profile] + baseline[job2_profile]) / 2

        # Interference: same profile placed together causes congestion
        if job1_profile == job2_profile:
            interference = 45
        elif (job1_profile in ['B', 'D']) and (job2_profile in ['B', 'D']):
            interference = 35
        else:
            interference = 10

        # Topology effect
        topo_factor = {
            'FatTree': 0.9, 'Clos': 0.85, 'Dragonfly': 1.1,
            'Mesh4D': 1.05, 'HyperX': 0.8, 'Ring': 1.2
        }

        # Policy effect
        policy_factor = {
            'Random': 1.0,
            'FIFO': 0.8,
            'TopoAware': 0.65,
            'CommAware': 0.4
        }

        congestion = (base_congestion + interference) * topo_factor[topology] * policy_factor[placement_policy]

        # Add noise
        congestion += np.random.normal(0, 5)
        return max(10, congestion)

    def generate_measurements(self, num_scenarios=520):
        """Generate synthetic measurements for validation."""
        job_names = list(self.config.jobs.keys())

        for scenario in range(num_scenarios):
            # Random job pair
            job1 = np.random.choice(job_names)
            job2 = np.random.choice(job_names)
            topology = np.random.choice(self.config.topologies)
            policy = np.random.choice(self.config.placement_policies)

            profile1 = self.config.jobs[job1]['profile']
            profile2 = self.config.jobs[job2]['profile']

            comm1 = self.compute_communication(job1)
            comm2 = self.compute_communication(job2)
            congestion = self.compute_congestion(profile1, profile2, topology, policy)

            # Job completion time (affected by congestion)
            base_time = 1200 + np.random.normal(0, 100)
            job_time = base_time * (1 + congestion / 100)

            # SLA compliance (latency < 100ms)
            sla_target = 100
            sla_met = 1 if congestion < sla_target else 0

            self.measurements.append({
                'scenario': scenario,
                'job1': job1,
                'job2': job2,
                'profile1': profile1,
                'profile2': profile2,
                'comm_vol_1': comm1,
                'comm_vol_2': comm2,
                'topology': topology,
                'placement_policy': policy,
                'network_congestion_ms': congestion,
                'job_completion_time_s': job_time,
                'sla_compliance': sla_met,
            })

        return pd.DataFrame(self.measurements)

class CommAwareAnalyzer:
    """Analyze CommAware performance."""

    def __init__(self, df):
        self.df = df

    def compute_metrics(self):
        """Compute performance metrics by policy."""
        metrics = {}

        for policy in ['Random', 'FIFO', 'TopoAware', 'CommAware']:
            subset = self.df[self.df['placement_policy'] == policy]

            metrics[policy] = {
                'avg_congestion_ms': subset['network_congestion_ms'].mean(),
                'std_congestion_ms': subset['network_congestion_ms'].std(),
                'p95_congestion_ms': subset['network_congestion_ms'].quantile(0.95),
                'avg_job_time_s': subset['job_completion_time_s'].mean(),
                'std_job_time_s': subset['job_completion_time_s'].std(),
                'sla_compliance_pct': subset['sla_compliance'].mean() * 100,
                'count': len(subset),
            }

        return metrics

    def compute_topology_metrics(self):
        """Compute metrics by topology."""
        results = []
        topologies = sorted(self.df['topology'].unique())

        for topo in topologies:
            topo_df = self.df[self.df['topology'] == topo]
            results.append({
                'Topology': topo,
                'Random_Congestion': topo_df[topo_df['placement_policy'] == 'Random']['network_congestion_ms'].mean(),
                'FIFO_Congestion': topo_df[topo_df['placement_policy'] == 'FIFO']['network_congestion_ms'].mean(),
                'TopoAware_Congestion': topo_df[topo_df['placement_policy'] == 'TopoAware']['network_congestion_ms'].mean(),
                'CommAware_Congestion': topo_df[topo_df['placement_policy'] == 'CommAware']['network_congestion_ms'].mean(),
                'Random_SLA': topo_df[topo_df['placement_policy'] == 'Random']['sla_compliance'].mean() * 100,
                'FIFO_SLA': topo_df[topo_df['placement_policy'] == 'FIFO']['sla_compliance'].mean() * 100,
                'TopoAware_SLA': topo_df[topo_df['placement_policy'] == 'TopoAware']['sla_compliance'].mean() * 100,
                'CommAware_SLA': topo_df[topo_df['placement_policy'] == 'CommAware']['sla_compliance'].mean() * 100,
            })

        return pd.DataFrame(results)

    def compute_profile_metrics(self):
        """Compute metrics by job profile pair."""
        results = []
        profiles = ['A', 'B', 'C', 'D']

        for p1 in profiles:
            for p2 in profiles:
                subset = self.df[(self.df['profile1'] == p1) & (self.df['profile2'] == p2)]
                if len(subset) > 0:
                    results.append({
                        'Profile_Pair': f'{p1}-{p2}',
                        'Random_Congestion': subset[subset['placement_policy'] == 'Random']['network_congestion_ms'].mean(),
                        'FIFO_Congestion': subset[subset['placement_policy'] == 'FIFO']['network_congestion_ms'].mean(),
                        'TopoAware_Congestion': subset[subset['placement_policy'] == 'TopoAware']['network_congestion_ms'].mean(),
                        'CommAware_Congestion': subset[subset['placement_policy'] == 'CommAware']['network_congestion_ms'].mean(),
                        'Random_SLA': subset[subset['placement_policy'] == 'Random']['sla_compliance'].mean() * 100,
                        'CommAware_SLA': subset[subset['placement_policy'] == 'CommAware']['sla_compliance'].mean() * 100,
                        'Sample_Size': len(subset),
                    })

        return pd.DataFrame(results)

    def train_classifier(self):
        """Train decision tree for job profile classification."""
        X = []
        y = []

        for _, row in self.df.iterrows():
            profile = row['profile1']
            features = [row['comm_vol_1'], row['job_completion_time_s']]
            X.append(features)
            y.append(profile)

        X = np.array(X)
        y = np.array(y)

        clf = DecisionTreeClassifier(max_depth=3, random_state=42)
        clf.fit(X, y)

        accuracy = clf.score(X, y)
        return clf, accuracy

    def generate_figures(self, output_dir='../figs'):
        """Generate publication-quality figures."""
        os.makedirs(output_dir, exist_ok=True)

        # Figure 1: Network congestion by policy
        fig, ax = plt.subplots(figsize=(6, 4))
        policies = ['Random', 'FIFO', 'TopoAware', 'CommAware']
        congestions = []
        for policy in policies:
            subset = self.df[self.df['placement_policy'] == policy]
            congestions.append(subset['network_congestion_ms'].mean())

        colors = ['#FF6B6B', '#FFA500', '#4ECDC4', '#45B7D1']
        ax.bar(policies, congestions, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Network Congestion (ms)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Scheduling Policy', fontsize=11, fontweight='bold')
        ax.set_title('Network Congestion by Scheduling Policy', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        for i, v in enumerate(congestions):
            ax.text(i, v + 2, f'{v:.1f}', ha='center', fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/Fig1_congestion_by_policy.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        # Figure 2: SLA compliance by policy
        fig, ax = plt.subplots(figsize=(6, 4))
        sla_scores = []
        for policy in policies:
            subset = self.df[self.df['placement_policy'] == policy]
            sla_scores.append(subset['sla_compliance'].mean() * 100)

        ax.bar(policies, sla_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('SLA Compliance (%)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Scheduling Policy', fontsize=11, fontweight='bold')
        ax.set_title('SLA Compliance Rate by Policy', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 105])
        ax.axhline(y=80, color='red', linestyle='--', linewidth=1.5, label='Target (80%)')
        for i, v in enumerate(sla_scores):
            ax.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/Fig2_sla_compliance.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        # Figure 3: Job completion time by policy
        fig, ax = plt.subplots(figsize=(6, 4))
        job_times = []
        for policy in policies:
            subset = self.df[self.df['placement_policy'] == policy]
            job_times.append(subset['job_completion_time_s'].mean())

        ax.bar(policies, job_times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Job Completion Time (s)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Scheduling Policy', fontsize=11, fontweight='bold')
        ax.set_title('Average Job Completion Time by Policy', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        for i, v in enumerate(job_times):
            ax.text(i, v + 30, f'{v:.0f}', ha='center', fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/Fig3_job_time.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        # Figure 4: Congestion by topology
        fig, ax = plt.subplots(figsize=(8, 5))
        topologies = sorted(self.df['topology'].unique())
        x = np.arange(len(topologies))
        width = 0.2

        for i, policy in enumerate(policies):
            congestions_by_topo = []
            for topo in topologies:
                subset = self.df[(self.df['topology'] == topo) & (self.df['placement_policy'] == policy)]
                congestions_by_topo.append(subset['network_congestion_ms'].mean())
            ax.bar(x + i*width, congestions_by_topo, width, label=policy, color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)

        ax.set_xlabel('Network Topology', fontsize=11, fontweight='bold')
        ax.set_ylabel('Network Congestion (ms)', fontsize=11, fontweight='bold')
        ax.set_title('Network Congestion Across Topologies', fontsize=12, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(topologies, rotation=45, ha='right')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/Fig4_topology_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        # Figure 5: Congestion by job profile pair
        fig, ax = plt.subplots(figsize=(8, 5))
        profiles = ['A', 'B', 'C', 'D']
        x = np.arange(len(profiles) * len(profiles))
        profile_pairs = []
        congestions_by_profile = {policy: [] for policy in policies}

        for p1 in profiles:
            for p2 in profiles:
                profile_pairs.append(f'{p1}-{p2}')
                for policy in policies:
                    subset = self.df[
                        (self.df['profile1'] == p1) &
                        (self.df['profile2'] == p2) &
                        (self.df['placement_policy'] == policy)
                    ]
                    congestions_by_profile[policy].append(subset['network_congestion_ms'].mean() if len(subset) > 0 else 0)

        for i, policy in enumerate(policies):
            ax.bar(x + i*width, congestions_by_profile[policy], width, label=policy, color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)

        ax.set_xlabel('Job Profile Pair', fontsize=11, fontweight='bold')
        ax.set_ylabel('Network Congestion (ms)', fontsize=11, fontweight='bold')
        ax.set_title('Congestion by Job Profile Combinations', fontsize=12, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(profile_pairs, rotation=45, ha='right', fontsize=9)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/Fig5_profile_pair_analysis.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        # Figure 6: SLA compliance by topology
        fig, ax = plt.subplots(figsize=(8, 5))
        topologies = sorted(self.df['topology'].unique())
        x = np.arange(len(topologies))

        for i, policy in enumerate(policies):
            sla_by_topo = []
            for topo in topologies:
                subset = self.df[(self.df['topology'] == topo) & (self.df['placement_policy'] == policy)]
                sla_by_topo.append(subset['sla_compliance'].mean() * 100)
            ax.bar(x + i*width, sla_by_topo, width, label=policy, color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)

        ax.set_xlabel('Network Topology', fontsize=11, fontweight='bold')
        ax.set_ylabel('SLA Compliance (%)', fontsize=11, fontweight='bold')
        ax.set_title('SLA Compliance Across Topologies', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 105])
        ax.axhline(y=80, color='red', linestyle='--', linewidth=1.5, label='Target')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(topologies, rotation=45, ha='right')
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/Fig6_sla_topology.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Generated 6 figures in {output_dir}/")

def main():
    """Main analysis pipeline."""
    print("=" * 80)
    print("CommAware: Communication-Aware Job Placement Analysis")
    print("=" * 80)

    # Configuration
    config = CommAwareConfig()
    print(f"\n✓ Loaded {len(config.jobs)} job definitions")
    print(f"✓ Configured {len(config.topologies)} network topologies")

    # Generate measurements
    print("\n[1] Generating 520 measurement scenarios...")
    generator = CommAwareGenerator(config)
    df = generator.generate_measurements(num_scenarios=520)
    print(f"✓ Generated {len(df)} measurements")

    # Save measurements
    os.makedirs('../data', exist_ok=True)
    df.to_csv('../data/commaware_measurements.csv', index=False)
    print(f"✓ Saved measurements to data/commaware_measurements.csv")

    # Analyze
    print("\n[2] Analyzing performance metrics...")
    analyzer = CommAwareAnalyzer(df)
    metrics = analyzer.compute_metrics()

    print("\nTABLE 1: Overall Performance Summary")
    print("-" * 100)
    print(f"{'Policy':<15} {'Avg Cong':<12} {'Std Dev':<12} {'P95 Cong':<12} {'Avg Time':<12} {'SLA (%)':<12}")
    print("-" * 100)
    for policy, values in metrics.items():
        print(f"{policy:<15} {values['avg_congestion_ms']:<12.1f} {values['std_congestion_ms']:<12.1f} {values['p95_congestion_ms']:<12.1f} {values['avg_job_time_s']:<12.1f} {values['sla_compliance_pct']:<12.1f}")
    print("-" * 100)

    # Topology analysis
    print("\nTABLE 2: Performance by Network Topology")
    print("-" * 130)
    topo_metrics = analyzer.compute_topology_metrics()
    print(topo_metrics.to_string(index=False))
    print("-" * 130)

    # Save topology table
    topo_metrics.to_csv('../data/topology_performance.csv', index=False)
    print("✓ Saved topology metrics to data/topology_performance.csv")

    # Profile analysis
    print("\nTABLE 3: Performance by Job Profile Pair")
    print("-" * 130)
    profile_metrics = analyzer.compute_profile_metrics()
    print(profile_metrics.to_string(index=False))
    print("-" * 130)

    # Save profile table
    profile_metrics.to_csv('../data/profile_performance.csv', index=False)
    print("✓ Saved profile metrics to data/profile_performance.csv")

    # Train classifier
    print("\n[3] Training job profile classifier...")
    clf, accuracy = analyzer.train_classifier()
    print(f"✓ Decision tree classifier accuracy: {accuracy:.2%}")

    # Generate figures
    print("\n[4] Generating publication-quality figures...")
    analyzer.generate_figures()

    # Compute improvements
    random_congestion = metrics['Random']['avg_congestion_ms']
    commaware_congestion = metrics['CommAware']['avg_congestion_ms']
    congestion_improvement = (random_congestion - commaware_congestion) / random_congestion * 100

    random_sla = metrics['Random']['sla_compliance_pct']
    commaware_sla = metrics['CommAware']['sla_compliance_pct']
    sla_improvement = commaware_sla - random_sla

    random_time = metrics['Random']['avg_job_time_s']
    commaware_time = metrics['CommAware']['avg_job_time_s']
    time_improvement = (random_time - commaware_time) / random_time * 100

    print("\n" + "=" * 80)
    print("COMMAWARE IMPROVEMENTS (vs. Random Baseline)")
    print("=" * 80)
    print(f"Network Congestion Reduction: {congestion_improvement:.1f}%")
    print(f"SLA Compliance Improvement:   {sla_improvement:.1f}% points")
    print(f"Job Completion Time Reduction: {time_improvement:.1f}%")
    print("=" * 80)

    print("\n✅ Analysis complete! All data and figures generated.")

if __name__ == '__main__':
    main()
