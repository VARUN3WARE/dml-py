"""
Tests for consistent error messages across PyDML.

Ensures all error messages follow consistent formatting:
- Lowercase messages (unless proper nouns)
- Include actual values received
- Include expected values/constraints
- Clear, actionable feedback
"""

import pytest
import torch
from pydml.utils.lr_scheduling import WarmupConfig, SchedulerConfig, SchedulerType
from pydml.losses.attention_transfer import AttentionTransferLoss
from pydml.strategies.curriculum import CurriculumStrategy
from pydml.strategies.peer_selection import DiversePeersSelector
from pydml.utils.ensemble import ensemble_predict
from pydml.utils.validation import validate_weights


class TestErrorMessageConsistency:
    """Test that all error messages are consistent and informative."""
    
    def test_warmup_config_errors_include_values(self):
        """Test WarmupConfig validation includes actual values."""
        with pytest.raises(ValueError) as exc_info:
            WarmupConfig(warmup_epochs=-5)
        assert "warmup_epochs must be non-negative, got -5" in str(exc_info.value)
        
        with pytest.raises(ValueError) as exc_info:
            WarmupConfig(warmup_start_lr=-0.1)
        assert "warmup_start_lr must be positive, got -0.1" in str(exc_info.value)
        
        with pytest.raises(ValueError) as exc_info:
            WarmupConfig(warmup_method='invalid')
        assert "warmup_method must be one of" in str(exc_info.value)
        assert "'invalid'" in str(exc_info.value)
    
    def test_scheduler_config_errors_show_valid_options(self):
        """Test scheduler errors show valid options."""
        from pydml.utils.lr_scheduling import create_scheduler_from_config
        
        # Test missing required parameters
        config = SchedulerConfig(
            scheduler_type=SchedulerType.MULTISTEP,
            base_lr=0.1
        )
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=0.1)
        
        with pytest.raises(ValueError) as exc_info:
            create_scheduler_from_config(optimizer, config)
        assert "milestones must be specified" in str(exc_info.value)
        assert "got None" in str(exc_info.value)
        
        config = SchedulerConfig(
            scheduler_type=SchedulerType.ONE_CYCLE,
            base_lr=0.1
        )
        with pytest.raises(ValueError) as exc_info:
            create_scheduler_from_config(optimizer, config)
        assert "total_steps must be specified" in str(exc_info.value)
        assert "got None" in str(exc_info.value)
    
    def test_attention_transfer_errors_include_counts(self):
        """Test AttentionTransferLoss errors include actual counts."""
        from pydml.losses.attention_transfer import MultiLayerAttentionTransferLoss
        loss_fn = MultiLayerAttentionTransferLoss()
        
        # Mismatched feature counts
        student_features = [torch.randn(4, 64, 32, 32) for _ in range(2)]
        teacher_features = [torch.randn(4, 64, 32, 32) for _ in range(3)]
        
        with pytest.raises(ValueError) as exc_info:
            loss_fn.forward(student_features, teacher_features)
        error_msg = str(exc_info.value)
        assert "got 2 student features" in error_msg
        assert "3 teacher features" in error_msg
        
        # Mismatched weights count
        loss_fn = MultiLayerAttentionTransferLoss(layer_weights=[0.5, 0.5, 0.5])
        student_features = [torch.randn(4, 64, 32, 32) for _ in range(2)]
        teacher_features = [torch.randn(4, 64, 32, 32) for _ in range(2)]
        
        with pytest.raises(ValueError) as exc_info:
            loss_fn.forward(student_features, teacher_features)
        error_msg = str(exc_info.value)
        assert "got 3 weights for 2 layers" in error_msg
    
    def test_attention_type_errors_show_valid_options(self):
        """Test attention type errors show valid options."""
        with pytest.raises(ValueError) as exc_info:
            loss_fn = AttentionTransferLoss(attention_type='invalid')
            loss_fn.compute_attention_map(torch.randn(4, 64, 32, 32))
        error_msg = str(exc_info.value)
        assert "unknown attention_type 'invalid'" in error_msg
        assert "must be one of" in error_msg
        assert "sum_squares" in error_msg
    
    def test_curriculum_strategy_errors_show_valid_options(self):
        """Test curriculum strategy errors show valid strategies."""
        strategy = CurriculumStrategy(strategy='invalid')
        
        with pytest.raises(ValueError) as exc_info:
            from torch.utils.data import TensorDataset
            dataset = TensorDataset(torch.randn(100, 3, 32, 32))
            dummy_model = torch.nn.Linear(10, 10)  # Dummy model
            strategy.create_curriculum_loader(dataset, model_or_models=dummy_model, batch_size=32)
        error_msg = str(exc_info.value)
        assert "unknown curriculum strategy 'invalid'" in error_msg
        assert "must be one of" in error_msg
        assert "confidence" in error_msg
        assert "loss" in error_msg
        assert "agreement" in error_msg
    
    def test_diversity_metric_errors_show_valid_options(self):
        """Test diversity metric errors show valid metrics."""
        from pydml.strategies.peer_selection import PeerSelectionConfig
        config = PeerSelectionConfig(strategy='diverse', k_peers=2, diversity_metric='invalid')
        selector = DiversePeersSelector(config)
        
        output1 = torch.randn(10, 10)
        output2 = torch.randn(10, 10)
        
        with pytest.raises(ValueError) as exc_info:
            selector._compute_diversity(output1, output2)
        error_msg = str(exc_info.value)
        assert "unknown diversity metric 'invalid'" in error_msg
        assert "must be one of" in error_msg
        assert "kl_div" in error_msg or "l2" in error_msg
    
    def test_ensemble_method_errors_show_valid_methods(self):
        """Test ensemble method errors show valid methods."""
        models = [torch.nn.Linear(10, 10) for _ in range(3)]
        inputs = torch.randn(4, 10)
        
        with pytest.raises(ValueError) as exc_info:
            ensemble_predict(models, inputs, method='invalid')
        error_msg = str(exc_info.value)
        assert "unknown ensemble method 'invalid'" in error_msg
        assert "must be one of" in error_msg
        assert "average" in error_msg
        assert "vote" in error_msg
    
    def test_weighted_ensemble_missing_weights_error(self):
        """Test weighted ensemble gives clear error when weights missing."""
        models = [torch.nn.Linear(10, 10) for _ in range(3)]
        inputs = torch.randn(4, 10)
        
        with pytest.raises(ValueError) as exc_info:
            ensemble_predict(models, inputs, method='weighted', weights=None)
        error_msg = str(exc_info.value)
        assert "weights must be provided" in error_msg
        assert "got None" in error_msg
    
    def test_validation_weights_all_zero_error(self):
        """Test weights validation gives clear error for all zeros."""
        with pytest.raises(ValueError) as exc_info:
            validate_weights([0.0, 0.0, 0.0], num_models=3)
        error_msg = str(exc_info.value)
        assert "at least one weight must be positive" in error_msg
        assert "got all zeros" in error_msg
    
    def test_error_messages_are_lowercase(self):
        """Test that error messages start with lowercase (except proper nouns)."""
        test_cases = [
            (lambda: WarmupConfig(warmup_epochs=-1), "warmup_epochs must"),
            (lambda: validate_weights([], 0), "at least one"),
        ]
        
        for func, expected_start in test_cases:
            try:
                func()
                pytest.fail(f"Expected ValueError for {func}")
            except ValueError as e:
                msg = str(e)
                # Check message starts with lowercase or is properly formatted
                assert msg[0].islower() or msg.split()[0] in ['None', 'True', 'False'], \
                    f"Error message should start with lowercase: '{msg}'"
    
    def test_error_messages_include_context(self):
        """Test that error messages provide helpful context."""
        # This should include what was expected and what was received
        with pytest.raises(ValueError) as exc_info:
            WarmupConfig(warmup_epochs=-5)
        
        error_msg = str(exc_info.value)
        # Should mention the parameter name
        assert "warmup_epochs" in error_msg
        # Should mention the constraint
        assert "non-negative" in error_msg
        # Should show the actual invalid value
        assert "-5" in error_msg
    
    def test_unknown_option_errors_show_valid_choices(self):
        """Test that 'unknown option' errors always show valid choices."""
        # All these should show what options ARE valid
        test_cases = [
            (lambda: AttentionTransferLoss(attention_type='bad').compute_attention_map(torch.randn(1, 1, 1, 1)), 
             ['sum_squares', 'mean_squares', 'mean_abs']),
            (lambda: CurriculumStrategy(strategy='bad').create_curriculum_loader(
                torch.utils.data.TensorDataset(torch.randn(10, 3, 32, 32)), None, 32),
             ['confidence', 'loss', 'agreement']),
        ]
        
        for func, expected_options in test_cases:
            try:
                func()
                pytest.fail(f"Expected ValueError for {func}")
            except ValueError as e:
                error_msg = str(e).lower()
                assert "must be one of" in error_msg or "unknown" in error_msg
                # Check that at least some valid options are mentioned
                for option in expected_options[:2]:  # Check first 2 options
                    assert option.lower() in error_msg


class TestErrorMessageFormat:
    """Test the formatting consistency of error messages."""
    
    def test_parameter_name_in_error(self):
        """Test that parameter name is always mentioned in error."""
        with pytest.raises(ValueError) as exc_info:
            WarmupConfig(warmup_epochs=-1)
        assert "warmup_epochs" in str(exc_info.value)
    
    def test_actual_value_in_error(self):
        """Test that actual invalid value is shown."""
        with pytest.raises(ValueError) as exc_info:
            WarmupConfig(warmup_epochs=-1)
        assert "-1" in str(exc_info.value)
    
    def test_constraint_description_in_error(self):
        """Test that constraint/expectation is described."""
        with pytest.raises(ValueError) as exc_info:
            WarmupConfig(warmup_epochs=-1)
        assert "non-negative" in str(exc_info.value) or "must" in str(exc_info.value)
    
    def test_actionable_error_messages(self):
        """Test that error messages are actionable (tell user what to do)."""
        with pytest.raises(ValueError) as exc_info:
            WarmupConfig(warmup_method='bad')
        
        error_msg = str(exc_info.value)
        # Should tell what IS valid
        assert "must be one of" in error_msg or "valid" in error_msg.lower()
