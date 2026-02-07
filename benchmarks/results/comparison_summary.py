"""
Quick Results Comparison

Comparison of baseline and DML experiments on CIFAR-10.
"""

# Baseline Results
print("=" * 60)
print("CIFAR-10 EXPERIMENT RESULTS COMPARISON")
print("=" * 60)
print()

print("BASELINE SINGLE MODELS:")
print("-" * 60)
print(f"{'Model':<20} {'Accuracy':<12} {'Params':<12} {'Time (min)':<12}")
print("-" * 60)
print(f"{'ResNet32':<20} {'93.50%':<12} {'467K':<12} {'59':<12}")
print(f"{'MobileNetV2':<20} {'92.50%':<12} {'2.2M':<12} {'224':<12}")
print(f"{'WRN-28-10':<20} {'96.05%':<12} {'36.5M':<12} {'1141':<12}")
print()

print("DEEP MUTUAL LEARNING (DML):")
print("-" * 60)
print(f"{'Configuration':<20} {'Accuracy':<12} {'Params':<12} {'Time (min)':<12}")
print("-" * 60)
print(f"{'2x ResNet32':<20} {'93.86%':<12} {'933K':<12} {'84':<12}")
print()

print("ANALYSIS:")
print("-" * 60)
print("DML 2x ResNet32 vs Single ResNet32:")
print(f"  Accuracy Improvement: +0.36% (93.50% → 93.86%)")
print(f"  Parameter Increase: 2x (467K → 933K)")
print(f"  Training Time Increase: +42% (59 min → 84 min)")
print()
print("Key Insights:")
print("  • DML achieves higher accuracy than single model")
print("  • Both DML models converged to similar accuracy (93.67%, 93.60%)")
print("  • Training overhead is modest (42% longer)")
print("  • Still more efficient than WRN-28-10 (39x fewer params)")
print()

print("Efficiency Comparison:")
print(f"  • ResNet32: 93.50% / 467K params = 0.200% per 1K params")
print(f"  • DML 2x ResNet32: 93.86% / 933K params = 0.101% per 1K params")
print(f"  • WRN-28-10: 96.05% / 36500K params = 0.003% per 1K params")
print()
print("  → Single ResNet32 is most parameter-efficient")
print("  → DML provides accuracy boost with reasonable cost")
print("  → WRN-28-10 achieves highest accuracy but very expensive")
print()

print("=" * 60)
