# tools.py
# Mock API tool for lead capture — triggered only after all three fields are collected.

import datetime


def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Mock API function to capture a qualified lead.
    In production, this would POST to a CRM like HubSpot, Salesforce, or a custom endpoint.

    Args:
        name:     Full name of the lead
        email:    Email address of the lead
        platform: Content platform (YouTube, Instagram, TikTok, etc.)

    Returns:
        Confirmation string
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*55}")
    print(f"  ✅  LEAD CAPTURED SUCCESSFULLY")
    print(f"{'='*55}")
    print(f"  Name     : {name}")
    print(f"  Email    : {email}")
    print(f"  Platform : {platform}")
    print(f"  Time     : {timestamp}")
    print(f"{'='*55}\n")

    return f"Lead captured successfully: {name}, {email}, {platform}"
