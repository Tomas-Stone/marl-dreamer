# Project Roadmap: Dreaming of Others 🧠

This document outlines the step-by-step implementation plan for the "Dreaming of Others" project, a multi-agent reinforcement learning agent based on world models. The goal is to build the architecture from scratch to gain a deep understanding of each component, starting with a single-agent Dreamer-style model and progressively extending it to our target architecture.

## Phase 1: Building a Single-Agent World Model (The "Dreamer" Core)

The first phase focuses on implementing a complete world model agent for a single-agent, image-based environment. This will serve as the foundation for our final MARL agent.

---

### **📍 Goal 0: Project Setup & Environment**

Before writing any model code, we need a solid foundation.

1.  **Framework:** Choose a primary deep learning framework.
    * **PyTorch:** Recommended for its user-friendly API and easier debugging.
    * **JAX:** For high performance, especially if leveraging existing libraries like `JaxMARL`.
2.  **Initial Environment:** Start with a simple, single-agent, image-based environment from `gymnasium`, such as `CarRacing-v2`. This allows for debugging the world model without multi-agent complexity.
3.  **Project Structure:** Create a clean directory structure:
    * `main.py`: Main script to run training.
    * `env.py`: Wrapper for environment handling.
    * `buffer.py`: Replay buffer implementation.
    * `models.py`: All neural network components (Encoder, RSSM, Actor, etc.).
    * `config.yaml`: Centralized file for hyperparameters.

---

### **🧩 Goal 1: The Building Block - A Variational Autoencoder (VAE)**

The stochastic component of the world model is essentially a VAE. Building one in isolation teaches the core concepts of latent variable modeling.

1.  **Implement the Encoder:** A CNN that takes an image and outputs the parameters (mean and standard deviation) of a Gaussian distribution.
2.  **Implement the Decoder:** A deconvolutional network that takes a sample from the latent space and reconstructs the original image.
3.  **Implement the VAE Loss:** Combine the **reconstruction loss** (`MSE` or `BCE`) and the **KL-divergence loss** to regularize the latent space.
4.  **Train and Validate:** Train the VAE on a dataset of images from the environment to ensure it can successfully compress and reconstruct observations.

> **Why this matters:** This step isolates the most conceptually difficult part of the world model—learning a compressed latent space and understanding the KL regularizer—before adding temporal dynamics.

---

### **📚 Goal 2: The Replay Buffer**

World models learn from sequences of experience, so the replay buffer must be designed to handle trajectories.

1.  **Data Structure:** Design the buffer to store and sample entire trajectories (e.g., sequences of 50-100 steps). Each step must store `(observation, action, reward, done_flag)`.
2.  **Sampling Logic:** Implement a function to randomly sample a batch of these trajectories for training the world model.
3.  **Interaction Loop:** Write the basic loop where a random agent populates the buffer by interacting with the environment.

---

### **❤️ Goal 3: The Heart - The Recurrent State-Space Model (RSSM)**

This is the core engine of the agent, combining a recurrent network (for memory) with a VAE-like structure (for uncertainty).

1.  **Transition Model (The Prior):** Implement the GRU component. It takes the previous state `(h_{t-1}, z_{t-1})` and action `a_{t-1}` to predict the *prior* distribution for the current latent state `z_t`.
2.  **Representation Model (The Posterior):** This part uses the GRU's output `h_t` and the current observation's embedding to compute the more accurate *posterior* distribution for `z_t`.
3.  **Forward Pass:** Implement the single-step logic: given the previous state, action, and current observation, compute the new state `(h_t, z_t)`.

---

### **🌍 Goal 4: The Complete Single-Agent World Model**

Assemble the pieces into a single, trainable model that learns the environment's dynamics.

1.  **Combine Components:** Create a `WorldModel` class containing the Encoder, RSSM, and Decoders (Image, Reward, Continue).
2.  **Implement the World Model Loss:** Write the full loss function which takes a sequence from the replay buffer and computes:
    * Image Reconstruction Loss
    * Reward Prediction Loss
    * Continue Flag Prediction Loss
    * Dynamics Loss (KL divergence between prior and posterior over the whole sequence).
3.  **Train and Debug:** Train the complete world model and log the individual loss components to ensure they are all decreasing as expected.

---

### **🤖 Goal 5: Imagination and the Actor-Critic**

With a trained world model, the agent can now learn to act by "dreaming."

1.  **Implement Actor & Critic:** Create two MLP networks that take the world model's state `(h_t, z_t)` as input to produce an action distribution and a value estimate.
2.  **Write the `imagine_ahead` Function:** This is the key to sample-efficient learning. The function takes a starting state and uses the (fixed) world model to unroll a long trajectory purely in latent space, using the Actor to select actions at each step.
3.  **Implement Actor-Critic Training:** Use the imagined trajectories to calculate lambda-returns and update the Actor and Critic networks to maximize future rewards.

> **🎉 Milestone:** You now have a working single-agent Dreamer-style agent! It should be capable of solving your initial test environment.

## Phase 2: Extending to MARL and Latent Teammate Modeling

With a solid single-agent foundation, we can now implement the novel architecture for multi-agent coordination.

---

### **🧑‍🤝‍🧑 Goal 6: Transition to MARL**

Adapt the existing codebase for the multi-agent setting.

1.  **Switch Environments:** Move to a MARL environment like `Overcooked-AI`, or environments from `pettingzoo` / `JaxMARL`.
2.  **Update Replay Buffer:** The buffer must now store the actions of *all* agents at each time step: `(x_t, a_t^0, a_t^j, r_t, c_t)`.
3.  **Adapt World Model Input:** The RSSM's transition model `h_t = f(...)` must now take the **joint action** `(a_{t-1}^0, a_{t-1}^j)` to predict the next state.

---

### **💡 Goal 7: Implement "Dreaming of Others"**

This is the final and most exciting step, implementing the core ideas from the paper.

1.  **Factorize the RSSM Latent State:** Modify the RSSM to output two separate latent variables, `z_t^{env}` and `z_t^{team}`. The KL divergence term in the loss will be the sum of the KL loss for each latent.
2.  **Add the ToM Head:** Implement the Teammate-Policy Decoder. This MLP takes `(h_t, z_t^{team})` as input and outputs a predicted action distribution for the teammate, `\hat{\pi}_t^j`.
3.  **Integrate the ToM Loss:** Add the `L_ToM` objective to the world model's training loss. This is a cross-entropy loss between the predicted teammate action distribution and the actual teammate action `a_t^j` from the replay buffer.
4.  **Update Actor-Critic Inputs:** Modify the Actor and Critic networks to take the full, factorized state `(h_t, z_t^{env}, z_t^{team})` as input.
5.  **Implement Social Imagination:** In the `imagine_ahead` function, at each imagined step:
    * The **Actor** samples the agent's action `a_t^0`.
    * The **ToM Head** samples a predicted teammate action `a_t^j`.
    * The world model uses the joint action `(a_t^0, a_t^j)` to predict the next state.

This roadmap provides a clear path from fundamental concepts to a cutting-edge research architecture. Happy coding!