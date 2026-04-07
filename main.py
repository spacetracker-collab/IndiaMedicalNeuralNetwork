import matplotlib.pyplot as plt
import numpy as np

def simulate_healthcare_training():
    """
    Simulates the 'training' of IndianHealthNet v15 from 1947 to 2050.
    """
    years = np.arange(1947, 2051)
    
    # 1. Mortality Loss (The Loss Function)
    # Starts high, drops sharply post-NRHM (2005), stabilizes in future.
    loss = 100 * np.exp(-0.025 * (years - 1947)) + 5 * np.random.normal(0, 0.1, len(years))
    
    # 2. Life Expectancy (The 'Accuracy' of the model)
    # Modeled as a logistic growth curve: L / (1 + exp(-k(x-x0)))
    accuracy = 32 + (40 / (1 + np.exp(-0.05 * (years - 1990))))
    
    # 3. OOPE vs Public Spend (The Gradient Noise)
    oope = 80 * np.exp(-0.01 * (years - 1947)) + 15 # Out of pocket stays high then drops
    public_spend = 1.0 + (1.5 / (1 + np.exp(-0.1 * (years - 2020)))) # GDP % target 2.5
    
    print("--- IndianHealthNet v15 Simulation Output ---")
    print(f"1947 Baseline: Life Exp: {accuracy[0]:.1f} | Loss: {loss[0]:.1f}")
    print(f"2026 Current:  Life Exp: {accuracy[79]:.1f} | Loss: {loss[79]:.1f}")
    print(f"2050 Forecast: Life Exp: {accuracy[-1]:.1f} | Loss: {loss[-1]:.1f}")
    print("---------------------------------------------")

    # Plotting the results
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('IndianHealthNet v15: System Performance (1947-2050)', fontsize=16)

    # Plot 1: Model Loss (Mortality)
    axs[0, 0].plot(years, loss, color='red', linewidth=2)
    axs[0, 0].set_title('System Loss (Mortality/Morbidity)')
    axs[0, 0].set_ylabel('Loss Index')
    axs[0, 0].grid(True, linestyle='--')

    # Plot 2: Model Accuracy (Life Expectancy)
    axs[0, 1].plot(years, accuracy, color='green', linewidth=2)
    axs[0, 1].set_title('Model Accuracy (Life Expectancy)')
    axs[0, 1].set_ylabel('Years')
    axs[0, 1].grid(True, linestyle='--')

    # Plot 3: Economic Gradient (OOPE)
    axs[1, 0].plot(years, oope, color='orange', label='Out-of-Pocket %')
    axs[1, 0].set_title('Systemic Noise (OOPE)')
    axs[1, 0].set_ylabel('Percentage of Total Spend')
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle='--')

    # Plot 4: Learning Rate (Public Spend GDP %)
    axs[1, 1].step(years, public_spend, color='blue', where='post')
    axs[1, 1].set_title('Learning Rate (Public Spend GDP %)')
    axs[1, 1].set_ylabel('GDP %')
    axs[1, 1].grid(True, linestyle='--')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    simulate_healthcare_training()
