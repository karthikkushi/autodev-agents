# Tip Calculator — Web App

## Overview
A tiny one-page website that splits a restaurant bill: enter the bill, pick a tip, choose how
many people, and instantly see the tip and what each person pays. Very small, but it must look
and feel premium — like an award-winning product page.

## Features
- Bill amount input
- Tip percentage: preset buttons 10 / 15 / 18 / 20 / 25 % plus a "Custom" field
- Number of people: − / + stepper (minimum 1)
- Results update as you type: tip total, grand total, tip per person, total per person
- Reset button that clears everything
- Remembers the last chosen tip % (localStorage)

## Tech Stack (keep it tiny)
- Python 3 + Flask
- `src/calculator.py` — one pure function that does the math, rounded to 2 decimals
- `src/app.py` — serves the page and one JSON endpoint
- `src/templates/index.html`, `src/static/style.css`, `src/static/app.js` — vanilla JS,
  no framework, no build step
- Run: `python src/app.py` then open http://localhost:5000

## API
- `POST /api/split` with `{"bill": 84.5, "tip_percent": 18, "people": 3}`
  → `{"tip_total": 15.21, "total": 99.71, "tip_per_person": 5.07, "total_per_person": 33.24}`
- Invalid input → status 400 with `{"error": "Enter a bill amount above 0", "field": "bill"}`
- The page calls this endpoint (debounced ~120 ms) so the math lives in one place.

## UI/UX Design
- One centered card (max width 440px) on a soft background, generous whitespace, one accent
  color, automatic dark mode (`prefers-color-scheme`), every color a CSS variable.
- Font: Inter (Google Fonts) with `system-ui` fallback; numbers use tabular numerals.
- The result panel sits on an accent-tinted surface; "per person" total is the hero (48px).
- Tip presets are a segmented button group with an obvious selected state.
- Validation shows inline under the field (never `alert()`); results show "—" until valid.
- Result numbers change with a subtle 150–200ms transition; respect `prefers-reduced-motion`.
- Mobile-first: works at 360px wide, touch targets at least 44px, `inputmode="decimal"` on
  the bill field.
- Accessible: a `<label>` for every input, visible focus rings, results in an
  `aria-live="polite"` region, fully usable with the keyboard.

## Tests
pytest for `calculator.py` (rounding, 0% tip, one vs several people, invalid input) and for
`POST /api/split` using Flask's test client.

## Edge Cases
- Empty, zero or negative bill → inline error, no results
- People below 1 → error; the stepper never goes below 1
- Custom tip above 100% or negative → error
- Per-person amounts rounded to 2 decimals
