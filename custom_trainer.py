import torch
from trl import GRPOTrainer

class AblationGRPOTrainer(GRPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss for Arm D: Per-answer normalization without standard deviation scaling.
        This isolates the length-bias mechanism from the success-probability reweighting.
        """
        rewards = inputs["rewards"]
        
        # Calculate mean for per-answer normalization
        mean_reward = rewards.mean(dim=-1, keepdim=True)
        
        # Calculate advantages WITHOUT standard deviation scaling
        # Standard GRPO would divide by std here
        advantages = rewards - mean_reward
        
        # Proceed with standard PPO/GRPO clipping mechanism using modified advantages
        # (This assumes integration with the rest of your training loop)
        
        return super().compute_loss(model, inputs, return_outputs)
