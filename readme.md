# Model Comparison Results

This table compares the **KL Divergence** and **MSE** values for different models. It helps evaluate the performance of each model in generating the **Up** and **Down** trajectories.

| Model                         | KL Divergence Up | MSE Up  | KL Divergence Down | MSE Down |
|-------------------------------|------------------|---------|--------------------|----------|
| **single_agent_dual_mode**     | 0.7255           | 2.7701  | -                  | -        |
| **single_agent_single_mode**   | 0.1974           | 0.7404  | -                  | -        |
| **two_agent_dual_mode**        | 0.7255           | 2.7700  | 9.7495             | 66.4397  |
| **two_agent_shared_mode**      | 0.0344           | 0.1922  | 0.0300             | 0.8005   |
| **two_agent_single_mode**      | 0.1974           | 0.7404  | 0.0304             | 0.7812   |
| **magail.py**                  | 0.1567           | 0.6616  | 0.0087             | 0.4258   |

### Explanation
- **KL Divergence** measures how much the generated trajectory distribution deviates from the expert data distribution. Lower values are better.
- **MSE (Mean Squared Error)** quantifies the average squared difference between the generated and expert trajectories. Again, lower values are better.
