# Russell 2000 Business Generator

## Overview

This script generates a realistic list of small businesses based on actual companies from the Russell 2000 index. The Russell 2000 is a stock market index that measures the performance of the smallest 2,000 companies in the Russell 3000 Index (companies ranked 1,001 to 3,000 by market capitalization), representing the small-cap segment of the U.S. equity universe.

## Features

- **Real Company Names**: Uses actual Russell 2000 company names and industries
- **Realistic Financial Attributes**: Generates industry-appropriate financial metrics including:
  - Annual revenue
  - Total debt
  - Credit scores
  - Google ratings and review counts
- **Multiple Industries**: Covers 10+ industry categories:
  - Healthcare
  - Technology
  - Retail
  - Food & Beverage
  - Construction
  - Personal Services
  - Financial Services
  - Industrial
  - Real Estate
  - Energy
- **Modular Design**: Easy to update company lists and generation parameters
- **Well-Documented**: Comprehensive docstrings and inline comments

## Company Data Source

The script includes a curated list of 100 actual Russell 2000 companies across diverse industries. This list is based on publicly available Russell 2000 index constituent data and can be easily updated or expanded.

## Usage

### As a Command-Line Script

Generate businesses directly from the command line:

```bash
# Generate default 50 businesses
python generate_russell2000_businesses.py

# Generate custom number of businesses
python generate_russell2000_businesses.py 100
```

The script will:
1. Generate the specified number of businesses
2. Display sample data and industry summary statistics
3. Save results to `russell2000_businesses.csv`

### As a Python Module

Import and use the functions in your own code:

```python
from generate_russell2000_businesses import generate_russell2000_businesses, get_industry_summary

# Generate 50 businesses
df = generate_russell2000_businesses(n=50, seed=42)

# View the data
print(df.head())
print(df.info())

# Get industry-level statistics
summary = get_industry_summary(df)
print(summary)
```

### Integration with LLM Loan Underwriting App

The script is integrated into the main application via `underwriting_agents_rag.py`:

```python
from generate_russell2000_businesses import generate_russell2000_businesses

# The existing generate_sample_businesses() function now uses Russell 2000 data
df = generate_sample_businesses(n=20)
```

## Output Format

The generated DataFrame includes the following columns:

- `name`: Company name (from Russell 2000)
- `industry`: Industry category (e.g., Healthcare, Technology)
- `sector`: Specific business sector (e.g., Cloud Communications)
- `revenue`: Annual revenue in dollars
- `debt`: Total debt in dollars
- `credit_score`: Business credit score (580-800 range)
- `google_rating`: Customer rating (1.0-5.0 scale)
- `review_count`: Number of customer reviews
- `debt_to_revenue_ratio`: Calculated debt-to-revenue ratio

## Industry Profiles

Each industry has realistic financial ranges based on small-cap business characteristics:

| Industry | Revenue Range | Debt Range | Credit Score Range |
|----------|--------------|------------|-------------------|
| Healthcare | $150K - $5M | $20K - $1.5M | 640-780 |
| Technology | $200K - $8M | $50K - $2M | 660-800 |
| Financial Services | $300K - $10M | $100K - $3M | 680-790 |
| Retail | $100K - $3M | $30K - $1.2M | 600-750 |
| Construction | $200K - $6M | $80K - $2.5M | 640-770 |
| Energy | $300K - $12M | $200K - $5M | 650-770 |

## Updating the Company List

The company list is stored in the `RUSSELL_2000_COMPANIES` list at the top of the script. To add new companies:

### Option 1: Edit the Script Directly

Add new company dictionaries to the list:

```python
RUSSELL_2000_COMPANIES = [
    # ... existing companies ...
    {"name": "New Company Inc", "industry": "Technology", "sector": "AI Services"},
    {"name": "Another Business", "industry": "Healthcare", "sector": "Telemedicine"},
]
```

### Option 2: Use the Helper Function

```python
from generate_russell2000_businesses import update_company_list

new_companies = [
    {"name": "New Company Inc", "industry": "Technology", "sector": "AI Services"},
    {"name": "Another Business", "industry": "Healthcare", "sector": "Telemedicine"},
]

update_company_list(new_companies)
```

Note: The helper function updates the list at runtime only. To persist changes, edit the script directly.

## Customization

### Change Industry Profiles

Edit the `INDUSTRY_PROFILES` dictionary to adjust financial ranges for each industry:

```python
INDUSTRY_PROFILES = {
    "Healthcare": {
        "revenue_range": (150_000, 5_000_000),
        "debt_range": (20_000, 1_500_000),
        "credit_score_range": (640, 780),
        # ... other ranges
    },
    # ... other industries
}
```

### Change Random Seed

For different random variations while maintaining reproducibility:

```python
df = generate_russell2000_businesses(n=50, seed=123)
```

## Dependencies

- `numpy`: For numerical operations and random number generation
- `pandas`: For data manipulation and DataFrame operations

These are already included in `requirements.txt`.

## Example Output

```
======================================================================
Russell 2000 Business Generator
======================================================================

Generating 50 businesses based on Russell 2000 companies...

✓ Generated 50 businesses
✓ Industries: Healthcare, Technology, Retail, Food & Beverage, Construction

Sample businesses (first 10):
----------------------------------------------------------------------
                    name   industry   revenue  credit_score  google_rating
            Amedisys Inc Healthcare  582867.0           731            4.7
   BioLife Solutions Inc Healthcare  566836.0           668            4.9
 Brookdale Senior Living Healthcare 3718398.0           740            3.7
...
```

## Future Enhancements

Potential improvements to consider:

1. **Real-time Data Integration**: Connect to financial APIs for live company data
2. **More Companies**: Expand the list to include all 2000 Russell index companies
3. **Additional Attributes**: Add more business metrics (employees, years in business, etc.)
4. **Industry-Specific Metrics**: Include sector-specific KPIs
5. **Geographic Data**: Add location information for companies
6. **Time Series**: Generate historical financial data

## License

This script is part of the AI Underwriting Demo project and follows the same MIT License.

## Support

For questions or issues:
1. Check the inline documentation in `generate_russell2000_businesses.py`
2. Review the example usage in this README
3. Open an issue in the repository
