# Model Comparison Results

This table compares the **KL Divergence** and **MSE** values for different models. It helps evaluate the performance of each model in generating the **Up** and **Down** trajectories.

| Model                         | KL Divergence Up(Single) | MSE Up(Single)  | KL Divergence Down | MSE Down |
|-------------------------------|------------------|---------|--------------------|----------|
| **single_agent_dual_mode**     | 279.5451           | 2.9626  | -                  | -        |
| **single_agent_single_mode**   | 0.3458           | 0.6673  | -                  | -        |
| **two_agent_dual_mode**        | 276.4294           | 2.7700  | 59.6717             | 66.4397  |
| **two_agent_shared_mode**      | 0.0265           | 0.1922  | 0.0607             | 0.8005   |
| **two_agent_single_mode**      | 0.3458           | 0.7404  | 0.0593             | 0.7812   |
| **magail.py**                  | 0.0185           | 0.6616  | 0.0381             | 0.4258   |

### Explanation
- **KL Divergence** measures how much the generated trajectory distribution deviates from the expert data distribution. Lower values are better.
- **MSE (Mean Squared Error)** quantifies the average squared difference between the generated and expert trajectories. Again, lower values are better.
