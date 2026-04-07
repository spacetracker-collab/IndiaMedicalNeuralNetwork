# 🧠 IndianHealthNet v15: Neural System Architecture

This repository contains the conceptual framework and simulation logic for **IndianHealthNet**, a 15-node deep neural model designed to simulate the evolution of the Indian Healthcare System from its post-independence inception (1947) to its predicted convergence (2030+).

---

## 🏗️ The 15-Node Neural Map

The model treats the national healthcare journey not as a timeline, but as a multi-layered network optimizing for **Universal Health Coverage (UHC)**.

### 🟢 Layer 0: The Input (Inception)
* **Node 1: Colonial Legacy (1947):** Initial weights biased toward urban centers and curative care.
* **Node 2: The Bhore Committee Bias:** Sets the activation function for state-sponsored, socialized medicine.

### 🟡 Layers 1-8: The Processing Core (Infrastructure & Economy)
* **Node 3: Sub-Center (SC):** The "Edge Node" of the network; low processing power but high physical reach.
* **Node 4: Primary Health Centre (PHC):** The local hub; processes diagnostic data from 6 SCs.
* **Node 5: Community Health Centre (CHC):** The "Switching Layer" for secondary specialty care.
* **Node 6: The GDP Gradient:** The learning rate bottleneck. Historically restricted to **~1.2%**, targeting **2.5%** for system convergence.
* **Node 7: OOPE (Out-of-Pocket Expenditure):** The system’s primary "Noise." High OOPE ($>60\%$) leads to signal degradation (poverty).
* **Node 8: The ASHA Optimizer:** A decentralized, human-intelligence layer designed to reduce the error rate in maternal and infant mortality.

### 🔵 Layers 9-12: The Variance (Education & Practice)
* **Node 9: Medical Education (MBBS):** High-weight nodes focused on clinical excellence.
* **Node 10: Dropout (Brain Drain):** Simulates the **30-40%** loss of trained human capital to international markets.
* **Node 11: Variant (AYUSH):** An ensemble layer integrating Ayurveda, Yoga, Unani, Siddha, and Homeopathy into the primary diagnostic stream.
* **Node 12: Digital Sandbox (ABDM):** The new data-bus node for interoperability and longitudinal health records.

### 🔴 Layers 13-15: Outcomes (The Output Layer)
* **Node 13: Historical Convergence:** Life expectancy delta from $32$ to $71+$ years.
* **Node 14: Predictive AI-Edge:** Future state where diagnostics (TB, NCDs, Imaging) move from the CHC to the PHC via AI.
* **Node 15: The Converged State (2030+):** The ultimate output—Universal Health Coverage with minimized systemic loss.

---

## 📈 System Metrics & Loss Functions

The model operates on a primary loss function designed to minimize national morbidity:

$$L(y, \hat{y}) = \sum (\text{IMR} + \text{MMR}) + \lambda(\text{OOPE})$$

Where:
* **IMR/MMR:** Infant/Maternal Mortality Rates.
* **$\lambda$:** The regularization parameter for financial protection.

### Predicted Delta (2026 ➡️ 2050)
| Metric | 1947 Baseline | 2026 Current | 2050 Predicted |
| :--- | :--- | :--- | :--- |
| **Life Expectancy** | 32 Years | 71.4 Years | 80+ Years |
| **Public Spend (GDP %)** | <0.5% | ~2.1% | 2.8% |
| **Digital Integration** | 0% | 45% (ABDM) | 98% |

---

## 🚀 Execution (main.py)
The accompanying `main.py` script visualizes these 15 nodes in action. It generates:
1.  **Loss Decay:** The reduction of mortality over time.
2.  **Accuracy Curve:** The rise of life expectancy as the model "learns."
3.  **The Gradient Gap:** Visualizing the relationship between GDP spend and Out-of-Pocket costs.

---

**Status:** `v15.main`
**Optimization:** Ongoing (Backpropagating ASHA feedback into the 2026 Policy layer).
**Satisfaction:** High (Clean architecture, zero bloat).
