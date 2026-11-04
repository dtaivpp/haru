"""Weather via Open-Meteo (https://open-meteo.com): free, no API key."""

from __future__ import annotations

import os

import httpx

from haru_gateway.tools import Tool, ToolContext, ToolError

USER_AGENT = "haru-gateway/0.1"

# WMO weather interpretation codes -> words Haru can say.
WMO = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "foggy", 48: "freezing fog",
       51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
       61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
       71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light showers", 81: "showers",
       82: "heavy showers", 85: "snow showers", 86: "heavy snow showers", 95: "thunderstorms",
       96: "thunderstorms with hail", 99: "thunderstorms with heavy hail"}


def weather_tools() -> list[Tool]:
    home = os.environ.get("HARU_HOME_LOCATION", "")
    units = os.environ.get("HARU_UNITS", "imperial")
    client = httpx.AsyncClient(timeout=10, headers={"User-Agent": USER_AGENT})

    async def geocode(place: str) -> dict:
        r = await client.get("https://geocoding-api.open-meteo.com/v1/search",
                             params={"name": place.split(",")[0].strip(), "count": 5, "language": "en"})
        r.raise_for_status()
        hits = r.json().get("results") or []
        if not hits:
            raise ToolError(f"couldn't find a place called {place!r}")
        # "Springfield, Illinois": prefer the hit whose region/country matches the rest.
        rest = [p.strip().lower() for p in place.split(",")[1:]]
        for h in hits:
            if all(any(p in str(h.get(k, "")).lower() for k in ("admin1", "country", "country_code")) for p in rest):
                return h
        return hits[0]

    async def get_weather(args: dict, ctx: ToolContext) -> dict:
        place = args.get("location") or home
        if not place:
            raise ToolError("no location given and no home location is configured; ask the user where")
        try:
            g = await geocode(place)
            imperial = units == "imperial"
            r = await client.get("https://api.open-meteo.com/v1/forecast", params={
                "latitude": g["latitude"], "longitude": g["longitude"], "timezone": "auto",
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m,precipitation",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,sunrise,sunset",
                "forecast_days": min(7, max(1, int(args.get("days", 7)))),
                "temperature_unit": "fahrenheit" if imperial else "celsius",
                "wind_speed_unit": "mph" if imperial else "kmh",
                "precipitation_unit": "inch" if imperial else "mm",
            })
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise ToolError(f"weather service unavailable: {exc}") from exc
        d = r.json()
        cur, daily = d.get("current", {}), d.get("daily", {})
        t_unit = "°F" if imperial else "°C"
        return {
            "place": ", ".join(p for p in (g.get("name"), g.get("admin1"), g.get("country")) if p),
            "now": {"conditions": WMO.get(cur.get("weather_code"), "unknown"),
                    "temperature": f"{cur.get('temperature_2m')}{t_unit}",
                    "feels_like": f"{cur.get('apparent_temperature')}{t_unit}",
                    "humidity": f"{cur.get('relative_humidity_2m')}%",
                    "wind": f"{cur.get('wind_speed_10m')} {'mph' if imperial else 'km/h'}"},
            "forecast": [{"date": day, "conditions": WMO.get(code, "unknown"),
                          "high": f"{hi}{t_unit}", "low": f"{lo}{t_unit}", "chance_of_precipitation": f"{pp}%"}
                         for day, code, hi, lo, pp in zip(daily.get("time", []), daily.get("weather_code", []),
                                                         daily.get("temperature_2m_max", []),
                                                         daily.get("temperature_2m_min", []),
                                                         daily.get("precipitation_probability_max", []))],
        }

    desc = "Current weather and a short forecast."
    desc += f" Defaults to the user's home ({home}) when no location is given." if home else \
        " No home location is configured, so always pass a location (ask the user if you don't know it)."
    return [Tool("get_weather", desc, {"type": "object", "properties": {
        "location": {"type": "string", "description": "City, optionally with region/country: \"Denver, Colorado\""},
        "days": {"type": "integer", "minimum": 1, "maximum": 7,
                 "description": "Forecast days including today, default 7 (enough for 'this weekend')"},
    }}, get_weather, timeout_s=15)]
