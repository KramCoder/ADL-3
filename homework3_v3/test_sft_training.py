#!/usr/bin/env python3
"""
Test script to verify SFT training works correctly and produces valid metrics.
"""
import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add the homework directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_sft_training():
    """Test that SFT training runs without errors and produces valid metrics."""
    import homework.sft as sft_module
    from transformers import TrainingArguments
    import numpy as np
    
    # Create a temporary output directory
    output_dir = tempfile.mkdtemp(prefix="sft_test_")
    
    try:
        print("=" * 60)
        print("Testing SFT Training Fix")
        print("=" * 60)
        
        # Patch TrainingArguments to limit training for testing
        original_init = TrainingArguments.__init__
        
        def patched_init(self, *args, **kwargs):
            # Add max_steps if not specified to limit training for testing
            if 'max_steps' not in kwargs:
                kwargs['max_steps'] = 20  # Limit to 20 steps
            if 'logging_steps' not in kwargs:
                kwargs['logging_steps'] = 5  # Log every 5 steps
            if 'save_strategy' not in kwargs:
                kwargs['save_strategy'] = 'no'  # Don't save during test
            return original_init(self, *args, **kwargs)
        
        TrainingArguments.__init__ = patched_init
        
        try:
            print(f"Output directory: {output_dir}")
            print("Running training (limited to 20 steps for testing)...")
            print("-" * 60)
            
            # This should not raise TypeError about num_items_in_batch
            sft_module.train_model(output_dir)
            
            print("\n" + "=" * 60)
            print("✅ Training completed without TypeError!")
            print("=" * 60)
            
            # Try to load and check metrics from tensorboard
            try:
                from tensorboard.backend.event_processing import event_accumulator
                ea = event_accumulator.EventAccumulator(output_dir)
                ea.Reload()
                
                scalars = ea.Tags().get('scalars', [])
                if scalars:
                    print("\nTraining Metrics:")
                    print("-" * 60)
                    for tag in scalars:
                        values = ea.Scalars(tag)
                        if values:
                            latest = values[-1]
                            print(f"{tag}: {latest.value:.6f} (step {latest.step})")
                            
                            # Validate metrics
                            if 'loss' in tag.lower():
                                if np.isnan(latest.value) or np.isinf(latest.value):
                                    print(f"  ❌ Invalid loss value!")
                                    return False
                                elif latest.value <= 0:
                                    print(f"  ⚠️  Warning: Loss <= 0")
                                else:
                                    print(f"  ✅ Loss is valid")
                            
                            if 'grad' in tag.lower() and 'norm' in tag.lower():
                                if np.isnan(latest.value) or np.isinf(latest.value):
                                    print(f"  ❌ Invalid grad norm!")
                                    return False
                                elif latest.value <= 0:
                                    print(f"  ⚠️  Warning: Grad norm <= 0")
                                else:
                                    print(f"  ✅ Grad norm is valid")
                            
                            if 'learning_rate' in tag.lower() or 'lr' in tag.lower():
                                if np.isnan(latest.value) or np.isinf(latest.value) or latest.value <= 0:
                                    print(f"  ❌ Invalid learning rate!")
                                    return False
                                else:
                                    print(f"  ✅ Learning rate is valid: {latest.value}")
                else:
                    print("\n⚠️  No scalar metrics found in tensorboard logs")
                    print("   (This might be okay if training was very short)")
                
            except Exception as e:
                print(f"\n⚠️  Could not read tensorboard logs: {e}")
                print("   (Training completed successfully, but couldn't verify metrics)")
            
            return True
            
        except TypeError as e:
            if 'num_items_in_batch' in str(e):
                print(f"\n❌ ERROR: The fix didn't work! Still getting TypeError:")
                print(f"   {e}")
                return False
            else:
                raise
        finally:
            # Restore original
            TrainingArguments.__init__ = original_init
        
    except Exception as e:
        print(f"\n❌ ERROR during training test: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)

if __name__ == "__main__":
    success = test_sft_training()
    sys.exit(0 if success else 1)
