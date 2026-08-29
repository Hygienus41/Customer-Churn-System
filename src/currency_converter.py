# ============================================================
# CURRENCY CONVERTER
# ============================================================

import requests
from datetime import datetime, timezone


# ============================================================
# BASE CURRENCY
# ============================================================

BASE_CURRENCY = "USD"


# ============================================================
# LIVE EXCHANGE RATE API
# ============================================================

EXCHANGE_RATE_API_URL = (
    "https://open.er-api.com/v6/latest/USD"
)

EXCHANGE_RATE_SOURCE = (
    "ExchangeRate-API Open Access"
)


# ============================================================
# FALLBACK EXCHANGE RATES
# ============================================================
# These rates are used if the live API cannot be reached.

FALLBACK_EXCHANGE_RATES = {
    "USD": 1.00,
    "NGN": 1339.76,
    "EUR": 0.85,
    "GBP": 0.74,
    "GHS": 12.50,
    "KES": 129.00,
    "ZAR": 17.30,
}


# ============================================================
# CURRENCY SYMBOLS
# ============================================================

CURRENCY_SYMBOLS = {
    "USD": "$",
    "NGN": "₦",
    "EUR": "€",
    "GBP": "£",
    "GHS": "₵",
    "KES": "KSh",
    "ZAR": "R",
}


# ============================================================
# LOAD LIVE EXCHANGE RATES
# ============================================================

def get_live_exchange_rates():
    """
    Retrieve the latest USD-based exchange rates
    from ExchangeRate-API Open Access.

    Returns
    -------
    tuple
        (exchange_rates, last_updated, status)

    If the API request fails, fallback rates are returned.
    """

    try:

        response = requests.get(
            EXCHANGE_RATE_API_URL,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        # Check whether the API request was successful
        if data.get("result") != "success":

            raise ValueError(
                "Exchange rate API returned an unsuccessful response."
            )

        live_rates = data.get(
            "rates"
        )

        if not live_rates:

            raise ValueError(
                "No exchange rates were returned by the API."
            )

        # Keep only currencies used by our application
        exchange_rates = {}

        for currency in FALLBACK_EXCHANGE_RATES:

            if currency in live_rates:

                exchange_rates[currency] = (
                    float(live_rates[currency])
                )

        # Make sure USD is always available
        exchange_rates["USD"] = 1.00

        # API provides the update timestamp
        last_updated_unix = data.get(
            "time_last_update_unix"
        )

        if last_updated_unix:

            last_updated = datetime.fromtimestamp(
                last_updated_unix,
                tz=timezone.utc
            ).strftime(
                "%B %d, %Y"
            )

        else:

            last_updated = (
                "Date unavailable"
            )

        return (
            exchange_rates,
            last_updated,
            "Live"
        )

    except Exception:

        return (
            FALLBACK_EXCHANGE_RATES.copy(),
            "Fallback rates",
            "Fallback"
        )


# ============================================================
# LOAD EXCHANGE RATES
# ============================================================

EXCHANGE_RATES, EXCHANGE_RATE_LAST_UPDATED, RATE_STATUS = (
    get_live_exchange_rates()
)


# ============================================================
# CONVERT CURRENCY
# ============================================================

def convert_currency(
    amount,
    from_currency=BASE_CURRENCY,
    to_currency=BASE_CURRENCY
):
    """
    Convert an amount from one currency to another.

    Parameters
    ----------
    amount : float
        Amount to convert.

    from_currency : str
        Currency of the original amount.

    to_currency : str
        Currency to convert into.

    Returns
    -------
    float
        Converted amount.
    """

    if from_currency not in EXCHANGE_RATES:

        raise ValueError(
            f"Unsupported source currency: {from_currency}"
        )

    if to_currency not in EXCHANGE_RATES:

        raise ValueError(
            f"Unsupported target currency: {to_currency}"
        )

    # Convert original currency to USD first
    amount_in_usd = (
        amount /
        EXCHANGE_RATES[from_currency]
    )

    # Convert USD to target currency
    converted_amount = (
        amount_in_usd *
        EXCHANGE_RATES[to_currency]
    )

    return converted_amount


# ============================================================
# GET CURRENCY SYMBOL
# ============================================================

def get_currency_symbol(currency):
    """
    Return the display symbol for a currency.
    """

    if currency not in CURRENCY_SYMBOLS:

        raise ValueError(
            f"Unsupported currency: {currency}"
        )

    return CURRENCY_SYMBOLS[currency]


# ============================================================
# GET EXCHANGE RATE INFORMATION
# ============================================================

def get_exchange_rate_info():
    """
    Return information about the current exchange rates.
    """

    return {
        "base_currency": BASE_CURRENCY,
        "source": EXCHANGE_RATE_SOURCE,
        "last_updated": EXCHANGE_RATE_LAST_UPDATED,
        "status": RATE_STATUS
    }