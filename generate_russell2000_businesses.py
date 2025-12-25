"""
Russell 2000 Business Generator

This module generates a realistic list of small businesses based on actual
Russell 2000 index companies. The Russell 2000 index represents approximately
2000 small-cap U.S. equities and serves as a benchmark for the small-cap segment
of the U.S. equity universe.

The script is modular and well-documented to allow easy updates to the company list
or generation parameters.

Usage:
    from generate_russell2000_businesses import generate_russell2000_businesses
    
    # Generate default set of businesses (50 companies)
    df = generate_russell2000_businesses()
    
    # Generate custom number of businesses
    df = generate_russell2000_businesses(n=100, seed=42)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional


# =====================================================
# RUSSELL 2000 COMPANY DATA
# =====================================================
# This is a representative sample of actual Russell 2000 companies
# with their real industries. This list can be updated as needed.
# Source: Publicly available Russell 2000 index constituent data

RUSSELL_2000_COMPANIES = [
    # Healthcare
    {"name": "Amedisys Inc", "industry": "Healthcare", "sector": "Healthcare Services"},
    {"name": "BioLife Solutions Inc", "industry": "Healthcare", "sector": "Medical Devices"},
    {"name": "Brookdale Senior Living", "industry": "Healthcare", "sector": "Healthcare Facilities"},
    {"name": "Cross Country Healthcare", "industry": "Healthcare", "sector": "Healthcare Staffing"},
    {"name": "Encompass Health Corp", "industry": "Healthcare", "sector": "Rehabilitation Services"},
    {"name": "LHC Group Inc", "industry": "Healthcare", "sector": "Home Healthcare"},
    {"name": "ModivCare Inc", "industry": "Healthcare", "sector": "Healthcare Services"},
    {"name": "Tenet Healthcare Corp", "industry": "Healthcare", "sector": "Hospital Systems"},
    
    # Technology
    {"name": "8x8 Inc", "industry": "Technology", "sector": "Cloud Communications"},
    {"name": "Blackline Inc", "industry": "Technology", "sector": "Financial Software"},
    {"name": "Box Inc", "industry": "Technology", "sector": "Cloud Storage"},
    {"name": "Clearwater Analytics", "industry": "Technology", "sector": "Investment Software"},
    {"name": "CommVault Systems", "industry": "Technology", "sector": "Data Management"},
    {"name": "Domo Inc", "industry": "Technology", "sector": "Business Intelligence"},
    {"name": "Dropbox Inc", "industry": "Technology", "sector": "Cloud Services"},
    {"name": "Everbridge Inc", "industry": "Technology", "sector": "Critical Communications"},
    {"name": "Five9 Inc", "industry": "Technology", "sector": "Contact Center Software"},
    {"name": "Guidewire Software", "industry": "Technology", "sector": "Insurance Software"},
    {"name": "HubSpot Inc", "industry": "Technology", "sector": "Marketing Software"},
    {"name": "Jamf Holding Corp", "industry": "Technology", "sector": "Device Management"},
    
    # Retail
    {"name": "Academy Sports & Outdoors", "industry": "Retail", "sector": "Sporting Goods"},
    {"name": "Abercrombie & Fitch", "industry": "Retail", "sector": "Apparel"},
    {"name": "American Eagle Outfitters", "industry": "Retail", "sector": "Apparel"},
    {"name": "Boot Barn Holdings", "industry": "Retail", "sector": "Western Wear"},
    {"name": "Buckle Inc", "industry": "Retail", "sector": "Denim & Casual Wear"},
    {"name": "Caleres Inc", "industry": "Retail", "sector": "Footwear"},
    {"name": "Chico's FAS Inc", "industry": "Retail", "sector": "Women's Apparel"},
    {"name": "Conn's Inc", "industry": "Retail", "sector": "Consumer Electronics"},
    {"name": "Designer Brands Inc", "industry": "Retail", "sector": "Footwear"},
    {"name": "Duluth Holdings Inc", "industry": "Retail", "sector": "Workwear"},
    
    # Food & Beverage
    {"name": "BJ's Restaurants", "industry": "Food & Beverage", "sector": "Casual Dining"},
    {"name": "Bloomin' Brands", "industry": "Food & Beverage", "sector": "Casual Dining"},
    {"name": "Brinker International", "industry": "Food & Beverage", "sector": "Casual Dining"},
    {"name": "Carrols Restaurant Group", "industry": "Food & Beverage", "sector": "Quick Service"},
    {"name": "Cheesecake Factory", "industry": "Food & Beverage", "sector": "Casual Dining"},
    {"name": "Cracker Barrel", "industry": "Food & Beverage", "sector": "Casual Dining"},
    {"name": "Dave & Buster's", "industry": "Food & Beverage", "sector": "Entertainment Dining"},
    {"name": "Denny's Corp", "industry": "Food & Beverage", "sector": "Family Dining"},
    {"name": "Del Taco Restaurants", "industry": "Food & Beverage", "sector": "Quick Service"},
    {"name": "First Watch Restaurant", "industry": "Food & Beverage", "sector": "Breakfast & Brunch"},
    
    # Construction
    {"name": "AECOM", "industry": "Construction", "sector": "Engineering Services"},
    {"name": "Aegion Corp", "industry": "Construction", "sector": "Infrastructure"},
    {"name": "Ameresco Inc", "industry": "Construction", "sector": "Energy Efficiency"},
    {"name": "Arcosa Inc", "industry": "Construction", "sector": "Construction Products"},
    {"name": "Comfort Systems USA", "industry": "Construction", "sector": "HVAC Services"},
    {"name": "Construction Partners", "industry": "Construction", "sector": "Infrastructure"},
    {"name": "Dycom Industries", "industry": "Construction", "sector": "Telecommunications"},
    {"name": "EMCOR Group", "industry": "Construction", "sector": "Building Services"},
    {"name": "Gibraltar Industries", "industry": "Construction", "sector": "Building Products"},
    {"name": "MYR Group Inc", "industry": "Construction", "sector": "Electrical Contracting"},
    
    # Personal Services
    {"name": "Amazing Lash Studio", "industry": "Personal Services", "sector": "Beauty Services"},
    {"name": "Bright Horizons", "industry": "Personal Services", "sector": "Childcare"},
    {"name": "European Wax Center", "industry": "Personal Services", "sector": "Beauty Services"},
    {"name": "Hand & Stone Franchise", "industry": "Personal Services", "sector": "Spa Services"},
    {"name": "H&R Block", "industry": "Personal Services", "sector": "Tax Preparation"},
    {"name": "Healthcare Services Group", "industry": "Personal Services", "sector": "Facility Services"},
    {"name": "Insperity Inc", "industry": "Personal Services", "sector": "HR Services"},
    {"name": "KBR Inc", "industry": "Personal Services", "sector": "Professional Services"},
    {"name": "Korn Ferry", "industry": "Personal Services", "sector": "Recruiting Services"},
    {"name": "Monro Inc", "industry": "Personal Services", "sector": "Automotive Services"},
    
    # Financial Services
    {"name": "Amalgamated Financial", "industry": "Financial Services", "sector": "Banking"},
    {"name": "Ameris Bancorp", "industry": "Financial Services", "sector": "Regional Banking"},
    {"name": "Atlantic Union Bankshares", "industry": "Financial Services", "sector": "Banking"},
    {"name": "Banc of California", "industry": "Financial Services", "sector": "Regional Banking"},
    {"name": "BancFirst Corp", "industry": "Financial Services", "sector": "Community Banking"},
    {"name": "Banner Corp", "industry": "Financial Services", "sector": "Regional Banking"},
    {"name": "Byline Bancorp", "industry": "Financial Services", "sector": "Community Banking"},
    {"name": "Cambridge Bancorp", "industry": "Financial Services", "sector": "Community Banking"},
    {"name": "Camden National Corp", "industry": "Financial Services", "sector": "Regional Banking"},
    {"name": "Cadence Bank", "industry": "Financial Services", "sector": "Regional Banking"},
    
    # Industrial
    {"name": "Alamo Group Inc", "industry": "Industrial", "sector": "Agricultural Equipment"},
    {"name": "Applied Industrial Tech", "industry": "Industrial", "sector": "Industrial Distribution"},
    {"name": "ArcBest Corp", "industry": "Industrial", "sector": "Logistics"},
    {"name": "Barnes Group Inc", "industry": "Industrial", "sector": "Industrial Components"},
    {"name": "Blue Bird Corp", "industry": "Industrial", "sector": "Bus Manufacturing"},
    {"name": "Chart Industries", "industry": "Industrial", "sector": "Industrial Equipment"},
    {"name": "Clean Harbors", "industry": "Industrial", "sector": "Environmental Services"},
    {"name": "Concrete Pumping", "industry": "Industrial", "sector": "Construction Equipment"},
    {"name": "Core & Main Inc", "industry": "Industrial", "sector": "Water Infrastructure"},
    {"name": "Donaldson Company", "industry": "Industrial", "sector": "Filtration Systems"},
    
    # Real Estate
    {"name": "American Campus Communities", "industry": "Real Estate", "sector": "Student Housing"},
    {"name": "Apartment Income REIT", "industry": "Real Estate", "sector": "Multifamily"},
    {"name": "Armada Hoffler Properties", "industry": "Real Estate", "sector": "Mixed-Use"},
    {"name": "Brandywine Realty Trust", "industry": "Real Estate", "sector": "Office"},
    {"name": "CareTrust REIT", "industry": "Real Estate", "sector": "Healthcare"},
    {"name": "Chatham Lodging Trust", "industry": "Real Estate", "sector": "Hospitality"},
    {"name": "Community Healthcare Trust", "industry": "Real Estate", "sector": "Medical Office"},
    {"name": "Corporate Office Properties", "industry": "Real Estate", "sector": "Office"},
    {"name": "CubeSmart", "industry": "Real Estate", "sector": "Self Storage"},
    {"name": "DiamondRock Hospitality", "industry": "Real Estate", "sector": "Hotels"},
    
    # Energy
    {"name": "Antero Resources", "industry": "Energy", "sector": "Oil & Gas Exploration"},
    {"name": "Archrock Inc", "industry": "Energy", "sector": "Gas Compression"},
    {"name": "Bristow Group", "industry": "Energy", "sector": "Helicopter Services"},
    {"name": "Callon Petroleum", "industry": "Energy", "sector": "Oil & Gas Production"},
    {"name": "ChampionX Corp", "industry": "Energy", "sector": "Oilfield Services"},
    {"name": "CNX Resources", "industry": "Energy", "sector": "Natural Gas"},
    {"name": "Comstock Resources", "industry": "Energy", "sector": "Oil & Gas"},
    {"name": "Core Laboratories", "industry": "Energy", "sector": "Reservoir Analysis"},
    {"name": "Delek US Holdings", "industry": "Energy", "sector": "Refining"},
    {"name": "Dril-Quip Inc", "industry": "Energy", "sector": "Drilling Equipment"},
]


# =====================================================
# INDUSTRY CHARACTERISTICS
# =====================================================
# Realistic financial ranges by industry based on small-cap business profiles

INDUSTRY_PROFILES = {
    "Healthcare": {
        "revenue_range": (150_000, 5_000_000),
        "debt_range": (20_000, 1_500_000),
        "credit_score_range": (640, 780),
        "rating_range": (3.5, 4.9),
        "review_count_range": (50, 2000),
    },
    "Technology": {
        "revenue_range": (200_000, 8_000_000),
        "debt_range": (50_000, 2_000_000),
        "credit_score_range": (660, 800),
        "rating_range": (3.8, 5.0),
        "review_count_range": (20, 1200),
    },
    "Retail": {
        "revenue_range": (100_000, 3_000_000),
        "debt_range": (30_000, 1_200_000),
        "credit_score_range": (600, 750),
        "rating_range": (3.0, 4.8),
        "review_count_range": (100, 3000),
    },
    "Food & Beverage": {
        "revenue_range": (120_000, 2_500_000),
        "debt_range": (40_000, 1_000_000),
        "credit_score_range": (620, 760),
        "rating_range": (3.2, 4.9),
        "review_count_range": (150, 4000),
    },
    "Construction": {
        "revenue_range": (200_000, 6_000_000),
        "debt_range": (80_000, 2_500_000),
        "credit_score_range": (640, 770),
        "rating_range": (3.5, 4.7),
        "review_count_range": (30, 800),
    },
    "Personal Services": {
        "revenue_range": (80_000, 2_000_000),
        "debt_range": (15_000, 800_000),
        "credit_score_range": (630, 760),
        "rating_range": (3.3, 4.9),
        "review_count_range": (80, 2500),
    },
    "Financial Services": {
        "revenue_range": (300_000, 10_000_000),
        "debt_range": (100_000, 3_000_000),
        "credit_score_range": (680, 790),
        "rating_range": (3.8, 4.8),
        "review_count_range": (40, 1000),
    },
    "Industrial": {
        "revenue_range": (250_000, 7_000_000),
        "debt_range": (100_000, 2_800_000),
        "credit_score_range": (650, 780),
        "rating_range": (3.6, 4.7),
        "review_count_range": (25, 600),
    },
    "Real Estate": {
        "revenue_range": (180_000, 8_000_000),
        "debt_range": (150_000, 4_000_000),
        "credit_score_range": (660, 790),
        "rating_range": (3.7, 4.8),
        "review_count_range": (50, 1500),
    },
    "Energy": {
        "revenue_range": (300_000, 12_000_000),
        "debt_range": (200_000, 5_000_000),
        "credit_score_range": (650, 770),
        "rating_range": (3.4, 4.6),
        "review_count_range": (20, 500),
    },
}


# =====================================================
# BUSINESS GENERATION FUNCTIONS
# =====================================================

def _generate_financial_attributes(
    industry: str,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Generate realistic financial attributes for a business based on its industry.
    
    Args:
        industry: The industry category of the business
        rng: NumPy random generator for reproducible randomization
    
    Returns:
        Dictionary containing revenue, debt, credit_score, google_rating, and review_count
    """
    profile = INDUSTRY_PROFILES.get(industry, INDUSTRY_PROFILES["Retail"])
    
    revenue = rng.integers(*profile["revenue_range"])
    debt = rng.integers(*profile["debt_range"])
    credit_score = rng.integers(*profile["credit_score_range"])
    rating = np.round(rng.uniform(*profile["rating_range"]), 1)
    review_count = rng.integers(*profile["review_count_range"])
    
    return {
        "revenue": float(revenue),
        "debt": float(debt),
        "credit_score": int(credit_score),
        "google_rating": float(rating),
        "review_count": int(review_count),
    }


def generate_russell2000_businesses(
    n: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a realistic list of businesses based on Russell 2000 index companies.
    
    The Russell 2000 is a small-cap stock market index that measures the performance
    of the smallest 2,000 companies in the Russell 3000 Index (companies ranked 1,001
    to 3,000 by market capitalization). This function uses real company names from the
    Russell 2000 and generates realistic financial attributes based on industry profiles.
    
    Args:
        n: Number of businesses to generate. If None, uses all available companies.
           If n > available companies, will cycle through the list.
        seed: Random seed for reproducible generation (default: 42)
    
    Returns:
        pandas.DataFrame with columns:
            - name: Company name (from Russell 2000)
            - industry: Industry category
            - sector: Specific business sector
            - revenue: Annual revenue in dollars
            - debt: Total debt in dollars
            - credit_score: Business credit score (580-800)
            - google_rating: Customer rating (1.0-5.0)
            - review_count: Number of customer reviews
    
    Example:
        >>> df = generate_russell2000_businesses(n=50)
        >>> print(df.head())
        >>> print(f"Industries: {df['industry'].unique()}")
    """
    rng = np.random.default_rng(seed)
    
    # Determine how many companies to generate
    available_companies = len(RUSSELL_2000_COMPANIES)
    if n is None:
        n = available_companies
    
    data = []
    
    for i in range(n):
        # Cycle through companies if n > available
        company = RUSSELL_2000_COMPANIES[i % available_companies]
        
        # Generate financial attributes based on industry
        financial_attrs = _generate_financial_attributes(company["industry"], rng)
        
        # Combine company info with financial attributes
        business_data = {
            "name": company["name"],
            "industry": company["industry"],
            "sector": company["sector"],
            **financial_attrs,
        }
        
        data.append(business_data)
    
    df = pd.DataFrame(data)
    
    # Add additional derived columns for analysis
    # Use a small epsilon to avoid division by zero
    df["debt_to_revenue_ratio"] = df.apply(
        lambda row: row["debt"] / max(row["revenue"], 1.0), axis=1
    )
    
    return df


def get_industry_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary statistics table grouped by industry.
    
    Args:
        df: DataFrame generated by generate_russell2000_businesses()
    
    Returns:
        DataFrame with industry-level summary statistics
    """
    summary = df.groupby("industry").agg({
        "name": "count",
        "revenue": ["mean", "median"],
        "debt": ["mean", "median"],
        "credit_score": ["mean", "median"],
        "google_rating": ["mean", "median"],
        "review_count": ["mean", "median"],
    }).round(2)
    
    summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]
    summary = summary.rename(columns={"name_count": "company_count"})
    
    return summary


def update_company_list(new_companies: List[Dict[str, str]]) -> None:
    """
    Helper function to update the RUSSELL_2000_COMPANIES list.
    
    This function can be used to add new companies to the data source.
    Note: This modifies the module-level constant at runtime but does not
    persist changes to the file.
    
    Args:
        new_companies: List of dictionaries with keys: name, industry, sector
    
    Example:
        >>> new_companies = [
        ...     {"name": "New Tech Corp", "industry": "Technology", "sector": "AI Services"},
        ...     {"name": "Regional Bank Co", "industry": "Financial Services", "sector": "Banking"},
        ... ]
        >>> update_company_list(new_companies)
    """
    global RUSSELL_2000_COMPANIES
    RUSSELL_2000_COMPANIES.extend(new_companies)
    print(f"Added {len(new_companies)} companies. Total: {len(RUSSELL_2000_COMPANIES)}")


# =====================================================
# COMMAND-LINE INTERFACE
# =====================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("Russell 2000 Business Generator")
    print("=" * 70)
    print()
    
    # Parse command line arguments
    n_companies = 50  # default
    if len(sys.argv) > 1:
        try:
            n_companies = int(sys.argv[1])
        except ValueError:
            print(f"Warning: Invalid number '{sys.argv[1]}', using default (50)")
    
    # Generate businesses
    print(f"Generating {n_companies} businesses based on Russell 2000 companies...")
    df = generate_russell2000_businesses(n=n_companies)
    
    print(f"\n✓ Generated {len(df)} businesses")
    print(f"✓ Industries: {', '.join(df['industry'].unique())}")
    print()
    
    # Display sample
    print("Sample businesses (first 10):")
    print("-" * 70)
    print(df[["name", "industry", "revenue", "credit_score", "google_rating"]].head(10).to_string(index=False))
    print()
    
    # Display industry summary
    print("\nIndustry Summary:")
    print("-" * 70)
    summary = get_industry_summary(df)
    print(summary.to_string())
    print()
    
    # Save to CSV
    output_file = "russell2000_businesses.csv"
    df.to_csv(output_file, index=False)
    print(f"✓ Saved to {output_file}")
    print()
    
    print("=" * 70)
    print("Script completed successfully!")
    print("=" * 70)
