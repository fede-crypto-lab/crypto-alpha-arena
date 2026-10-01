# Dati pubblici salvati nel repository

Solo dati di **pubblico dominio** (governo USA). Il repository è pubblico: i dati
con licenza (Databento/CME, TradingView, SeasonAlgo, S&P 500, VIX) **non** vanno
messi qui — restano in `data/` (ignorata da git) o in un archivio privato.

| file | contenuto | fonte | scaricato |
|---|---|---|---|
| `fred/DTB3.csv` | BOT USA 3 mesi, giornaliero | FRED (US Treasury) | 2026-10-01 |
| `fred/DGS2.csv` | Treasury 2 anni | FRED (US Treasury) | 2026-10-01 |
| `fred/DFII10.csv` | tasso reale 10 anni (TIPS) | FRED (US Treasury) | 2026-10-01 |
| `fred/DTWEXBGS.csv` | dollaro USA ponderato (broad) | FRED (Federal Reserve) | 2026-10-01 |
| `fred/SOFR.csv` | SOFR | FRED (NY Fed) | 2026-10-01 |
| `eia/eia_ng_storage_weekly.xls` | scorte di gas Lower 48, settimanale dal 2010 | EIA | 2026-10-01 |
| `usda/psd_grains_pulses_csv.zip` | USDA PSD cereali (valori finali, rivisti) | USDA FAS | 2026-09 |
| `usda/wasde_2010-2026_as_published.tar.gz` | 198 report WASDE **così come pubblicati** (testo; Excel appiattito per 2011-2016) + `index.json` | USDA ESMIS | 2026-09-11 |

Note:
- Nei backtest usare il WASDE "as published": il PSD contiene valori rivisti dopo
  la pubblicazione, che leggerebbero il futuro.
- EIA: la settimana finisce il venerdì e il dato esce il giovedì dopo
  (`scarcity.EIA_RELEASE_LAG_DAYS`).
- Uso: `tar -xzf usda/wasde_2010-2026_as_published.tar.gz -C <cartella>` e poi
  `scarcity.load_wasde('<cartella>/wasde')`; `scarcity.parse_eia_storage(open(xls,'rb').read())`.
