# PriceSense – Dynamic Pricing & Demand Elasticity Analyzer

## Business Problem
Retail organizations frequently struggle to balance volume growth with profit margins because they apply blanket pricing strategies across an entire customer base. Without understanding how different customer segments react to price changes (price elasticity), companies either leave money on the table by underpricing inelastic buyers or destroy sales volume by overpricing highly sensitive shoppers. **PriceSense** solves this by using transaction data to calculate segment-level price elasticity, simulating conjoint analysis to identify purchase drivers, and modeling clear financial scenarios to recommend precise, risk-adjusted pricing actions.

## Technology Stack
- **Languages**: Python 3
- **Data Manipulation**: Pandas, NumPy
- **Statistical Modeling**: Statsmodels (OLS Regression), Scikit-Learn (Linear Regression, Label Encoding)
- **Data Visualization**: Matplotlib, Seaborn
- **Storage**: MySQL (Primary), SQLite (Fallback)
- **Environment**: Jupyter Notebook

---

## Tableau Dashboard

[![PriceSense Tableau Dashboard](outputs/PriceSense%20Dashboard.png)](https://public.tableau.com/)

**[View Interactive Dashboard on Tableau Public](https://public.tableau.com/)** *(Placeholder Link)*

## Tableau Dashboard Guide
The following CSV outputs are pre-formatted for direct ingestion into Tableau to build an interactive Pricing Strategy Dashboard.

### 1. Elasticity Heatmap
- **Source File**: `elasticity_by_segment.csv`
- **Purpose**: Displays the price sensitivity of each customer tier.
- **Fields to Use**: `segment` (Rows), `elasticity` (Color / Text Label), `interpretation` (Tooltip).
- **Format**: Build a text table or heatmap. Color scale: Red for Highly Elastic (<-1.5), Blue for Inelastic (>-0.7).

### 2. Regional Elasticity Map
- **Source File**: `elasticity_by_region.csv`
- **Purpose**: Geographically maps which markets can sustain price increases versus those needing discounts.
- **Fields to Use**: `region` (Geographic Role: Country), `elasticity` (Color).
- **Format**: Filled Map (Choropleth).

### 3. Attribute Importance Chart
- **Source File**: `conjoint_importance.csv`
- **Purpose**: Shows what factors (e.g., direct price vs. competitor gap vs. category) drive purchase decisions most strongly.
- **Fields to Use**: `attribute` (Rows), `importance_score` (Columns).
- **Format**: Horizontal Bar Chart sorted descending by score.

### 4. Scenario Impact Panel
- **Source File**: `scenario_results.csv`
- **Purpose**: A "what-if" simulator showing the financial impact of distinct pricing changes per segment.
- **Fields to Use**: `segment` (Columns), `price_change_label` (Rows), `margin_delta` (Text / Color).
- **Filters**: Create an interactive filter on `recommendation` to only show "Recommended" or "Caution" paths.
- **Format**: Highlight Table formatting Margin Delta with a diverging Red-Green color palette.

### 5. Price Response Curves
- **Source File**: `price_response_curves.csv`
- **Purpose**: Visualizes the continuous volume drop-off as prices increase.
- **Fields to Use**: `price_index` (Columns, Treat as Continuous Dimension), `predicted_quantity` (Rows), `segment` (Color).
- **Format**: Multi-line chart with a dual axis tracking `revenue` on a dotted secondary line.
# PriceSense
