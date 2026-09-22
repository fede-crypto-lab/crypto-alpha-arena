# MEMORIA — fatti stabiliti, da non rimisurare

Tabella di consultazione, non narrativa. La spiegazione di *come* ci sono arrivato
sta in `README.md`; qui ci sono solo i risultati, i vicoli ciechi e le condizioni
al contorno, perché una sessione nuova non rifaccia lavoro già fatto.

Ogni numero qui è stato **misurato**, non stimato. Dove una cosa è un'ipotesi non
verificata, è scritto esplicitamente.

Ultimo aggiornamento: sessione del 2026-09-22, branch
`claude/arbitrage-bot-analysis-4b7vkj`.

---

## 1. Verdetto attuale in una riga

Il carry sul funding (long spot / short perp, selezione trasversale) è un edge
**reale, strutturale e non direzionale**, che vale **+0.96% APR sul capitale**
misurato su 540 giorni con basis reale e universo liquido. È sotto il risk-free.
Ogni volta che ho sostituito un'ipotesi con una misura, la stima è scesa.

---

## 2. Strade già percorse — NON ripercorrerle

| strada | esito | evidenza |
|---|---|---|
| Spread funding cross-venue (HL↔OKX) | **morta** | Spread medio ~5% APR contro 27bp round trip. Sweep 5 soglie × 3 hold × 2 asset: ogni configurazione con ≥1 trade ha expectancy negativa |
| Timing temporale su singolo asset (EWMA) | **morta** | Entry selectivity **0.83** (sotto 1 = entra quando le condizioni sono peggiori della media). La versione passiva rende **9×** quella selettiva |
| Indicatori di prezzo come segnale di funding | **morta** | IC vs funding futuro: MACD 0.025, momentum −0.051, SMA200 −0.067, volatilità 0.046. Il funding stesso: **0.498** |
| Long-short trasversale direzionale | **scartata a ragion veduta** | Cattura l'intero spread Q1−Q5 (~19-30% APR) ma introduce rischio di prezzo relativo. Confermata da terzi: nsheng1568 riporta ~7% APR a ~25% vol, Sharpe ~0.28 |

---

## 3. Persistenza del funding — la premessa, validata

Correlazione di rango di Spearman fra funding trailing e funding realizzato nel
periodo successivo, **finestre non sovrapposte**.

**3 anni, 39 coin, Hyperliquid:**

| finestra | periodi | ρ medio | ρ min | top quintile | mediana | bottom quintile |
|---|---|---|---|---|---|---|
| 7g | **155** | 0.652 | +0.21 | +26.7% APR | +14.8% | −3.8% |
| 14g | 77 | 0.648 | +0.26 | +25.6% | +15.1% | −1.1% |
| 30g | 35 | 0.633 | +0.34 | +23.4% | +15.8% | +0.4% |
| 90g | 11 | 0.643 | +0.53 | +20.7% | +13.0% | +2.2% |

**Mai negativa in 155 periodi indipendenti.** Su 180 giorni i valori sono
praticamente identici (0.669 / 0.649 / 0.672 / 0.639), quindi non è un artefatto
della finestra.

Il funding storico è il **doppio** di quello recente: top quintile +26.7% su 3
anni contro +12.1% negli ultimi 180 giorni. La finestra recente è il periodo
*magro*, non quello favorevole.

---

## 4. Risultati dei backtest

| configurazione | campione | basis | netto | APR | liq. | win rate [CI 95% low] |
|---|---|---|---|---|---|---|
| carry passivo BTC 1× (OKX) | 96g | −$1 | +$103 | +1.97% | 0 | — |
| trasversale, slippage piatto 5bp | 180g | +$14 | +$1.398 | +2.84% | 3 | 100% [78.5%] |
| trasversale, slippage **misurato** | 180g | +$71 | +$1.074 | +2.18% | 3 | 83.3% [**55.2%**] |
| **MEXC 540g, solo liquide** | **540g** | **−$70** | **+$1.424** | **+0.96%** | 3 | 75% [**55.1%**] |
| MEXC 540g, con alt sottili | 540g | **−$1.322** | −$283 | −0.19% | 7 | 68.8% [44.4%] |

Notional 10.000 $/gamba, leva 1×, capitale = 2× notional.

**Robustezza:** sweep su 144 configurazioni (slippage × slot × hold × isteresi ×
lookback × frequenza): **141 positive**. Degrado monotono con lo slippage —
2bp: 48/48, mediana +1.8%; 5bp: 48/48, mediana +1.4%; 10bp: 45/48, mediana +0.8%.

**Driver strutturali** (tutti coerenti con "non fare churn"):
- isteresi larga (exit_rank 20 vs 12): 10.6 rotazioni vs 14.7, utilizzo 60% vs 45%, APR +1.79% vs +1.19%
- lookback 168h vs 336h: utilizzo 67% vs 38%, APR +1.88% vs +1.10%

---

## 5. Costo di esecuzione — misurato dai book reali

Round trip di un carry (4 attraversate), $10.000/gamba, snapshot live HL+OKX:

| BTC | ETH | HYPE | ZEC | TAO | AAVE | LIT | NEAR | ONDO | ENA | XPL | PUMP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1bp | 1.1bp | 1.9bp | 2.9bp | 10.1bp | 15.9bp | 22.9bp | 27.6bp | 38.9bp | 40.1bp | 80.3bp | **488.4bp** |

**Rapporto 4.000× fra la più economica e la più cara.** Il ranking trasversale
seleziona l'estremo caro, perché un tasso è alto in parte *perché* nessuno vuole
stare dall'altra parte.

**Coda (archivi Binance bookDepth, 14 giorni, 40.320 snapshot/simbolo):**

| simbolo | mediana | p99 | **p99/mediana** |
|---|---|---|---|
| NEAR | 0.9bp | 1.6bp | 1.8× |
| WLD | 1.4bp | 2.6bp | 1.9× |
| AAVE | 1.5bp | 9.8bp | 6.6× |
| ONDO | 1.1bp | 80.4bp | 71× |
| ENA | 1.4bp | 277bp | **193×** |
| TAO | 1.0bp | 295bp | **296×** |

La fragilità del book è **indipendente** dalla liquidità mediana. Su mediane quasi
identiche, le code differiscono di due ordini di grandezza. Il filtro va sul p99.

---

## 6. Qualità della copertura (basis)

Movimento del basis perp-vs-spot su hold di 20 giorni, MEXC, 540 giorni. Quello
che finisce nel PnL è il **cambiamento**, non il livello (un premio costante non
costa niente).

22 coin su 23 hanno p99 sotto l'1%. Le peggiori: HYPE 1.66%, LIT 0.95%, VVV 0.83%,
XPL 0.43%. Le migliori: ETH 0.07%, SOL/XRP/DOGE/LTC 0.10-0.16%.

**Meccanismo scoperto (non un bug — verificato a mano, invariante delta=0 su 16/16
trade):** la copertura a notional fisso si sfalda quando il prezzo corre. Trade
peggiore, PONS: rapporto perp/spot mosso del **6.5%**, perdita del **12.6%**,
perché il token è quasi raddoppiato durante l'hold e le gambe si sono sbilanciate
di taglia. È **l'assenza di ribilanciamento**, non un difetto contabile.

Corollario importante: **le perdite di basis sono concentrate nei trade liquidati.**
Ogni trade chiuso per `max_hold` o `end_of_sample` ha basis positivo o nullo.
Liquidazione e perdita di basis sono lo stesso evento visto da due angoli.

---

## 7. Volatilità come cancello di rischio (non come segnale)

IC della volatilità trailing 168h:
- vs funding futuro: **0.046** (inutile)
- vs rialzo avverso che liquida la gamba short: **0.356**
- vs movimento del basis: **0.363**

Quintili di volatilità, 1.514 osservazioni, rialzo massimo su hold di 20 giorni:

| quintile | mediana | p99 | **oltre +99.5% = liquidazione a 1×** |
|---|---|---|---|
| Q1 (calmo) | 7.5% | 63.4% | **0.00%** |
| Q2 | 10.0% | 84.3% | 0.34% |
| Q3 | 11.6% | 94.5% | 0.68% |
| Q4 | 15.1% | 166.0% | 3.04% |
| Q5 (agitato) | 18.5% | 157.2% | **5.76%** |

Implementato in `PortfolioParams.exclude_vol_quantile`. Effetto misurato:
**liquidazioni da 4 a 0**, ma **utilizzo dal 22% al 5%**. Riduce il rischio di
coda; non fa guadagnare di più. Il miglioramento di APR nello sweep del gate NON
va creduto (5-10 rotazioni, soglia scelta in-sample).

---

## 8. Costi futures IBKR — calcolati, non misurati sul campo

Spread calendario = 4 fill + attraversamento spread. Prezzi FRED 2026-09:
WTI 107.02, S&P 7764.70.

| strumento | notional | round trip | breakeven a 14g |
|---|---|---|---|
| ES (S&P e-mini) | $388.235 | 0.87bp | **0.46%** |
| MES (S&P micro) | $38.824 | 1.28bp | 0.67% |
| CL (WTI) | $107.020 | 2.69bp | 1.40% |
| MCL (WTI micro) | $10.702 | 4.19bp | 2.18% |
| *carry cripto MEXC* | *$10.000* | *34.0bp* | ***8.86%*** |

**I futures costano 4-20× meno del cripto.** A parità di size ($10.702 su MCL
contro $10.000 su MEXC) sono comunque 8× più economici. Questo ribalta
l'aritmetica che ha limitato tutto il lavoro sul cripto.

Assunzioni: commissione ES $2.24/fill, micro $0.62/fill, 1 tick di spread
attraversato per lato. **Da verificare sui fill reali.**

---

## 9. Raggiungibilità delle fonti dati (da container cloud)

| fonte | stato | profondità |
|---|---|---|
| Hyperliquid `/info` | ✅ | funding **3 anni**; candele **208g** (cap del venue) |
| OKX | ✅ | funding ~96g; candele spot 208g, perp 96g |
| MEXC | ✅ **richiede UA browser + rate limit** | funding **540g**; candele perp ~1000g, spot **3 anni** |
| KuCoin / Coinbase / Kraken / Gate / Bitget spot | ✅ | anni |
| Binance `data.binance.vision` (bookDepth) | ✅ | **dal 2023**, ogni 30s, tutti i simboli |
| FRED | ✅ | serie macro, niente curve futures |
| Bybit API | ❌ | CloudFront blocca per regione |
| Binance REST API | ❌ | geo-restricted |
| CBOE (storico VIX) | ❌ | 403 |
| Stooq | ❌ | challenge JavaScript |
| Yahoo Finance | ❌ | rate-limited |
| Nasdaq Data Link | ❌ | 403 |
| TAAPI | ⚠️ | ha `/candles`, **non ha endpoint funding** (verificato: "does not exist") |

### 9-bis. Curve futures — cosa è raggiungibile

| fonte | stato | copertura |
|---|---|---|
| **EIA** `eia.gov/dnav/{pet,ng}/hist_xls/` | ✅ | Contratti 1-4: **WTI 1983-2024** (10.183 oss), **gas 1994-2024** (7.496), heating oil (9.894), benzina (4.609). Fino al **2024-04-05**; l'API v2 richiede una chiave gratuita per i dati correnti |
| CME settlements | ❌ | Blocca esplicitamente lo scraping e lo vieta nei termini d'uso. **Non aggirare** |
| Barchart, investing.com | ❌ | 403 |
| IBKR/TWS | 🔑 | La fonte pulita per un universo ampio: richiede l'account dell'utente |

Con l'EIA si fa il test di decomposizione su 4 commodity energetiche subito e
gratis. Per il test **trasversale** servono 12-20 commodity di settori diversi, e
lì serve IBKR.

**Conseguenza per la fase futures: lo storico delle curve non è ottenibile
gratuitamente da un container cloud.** La fonte pulita è l'account IBKR.

---

## 10. Bug trovati — ognuno ha un test di regressione

| # | bug | perché era pericoloso |
|---|---|---|
| 1 | `spearman` non gestiva i pareggi | Il funding HL ha un floor dove decine di coin stanno identiche: ranghi arbitrari avrebbero fabbricato accordo. *Corretto: risultato invariato (0.669 vs 0.670)* |
| 2 | Book troncati a 20 livelli su 400 | Su PUMP mostrava $0 di profondità invece di $412.000 |
| 3 | Parser bookDepth: campo `"-5.00"` letto come `int` | Scartava **tutte** le 2.880 righe in silenzio |
| 4 | Archivio Binance con giornate corrotte | NEAR riportava $13 identici su tutte le bande → falso "30% non assorbe $10k" |
| 5 | Throttling MEXC scartava coin dall'universo | BTC, AAVE, BCH, DOT spariti da un run: non riproducibile e distorto |
| 6 | Leva applicata anche alla gamba spot | Inventava liquidazioni impossibili e sottostimava il capitale |
| 7 | `--max-hold` default 21g in modalità portafoglio | Uscita a timer su un book che deve uscire per rango: costo puro |

**Pattern:** i bug 1, 3, 4, 5 producevano tutti numeri *plausibili* invece di
errori. È il motivo per cui ogni modulo ha test che verificano invarianti, non
solo assenza di eccezioni.

---

## 11. Lavoro correlato — già cercato, non rifarlo

| progetto | cosa fa | risultato |
|---|---|---|
| [aaronpascalkujur/trading-strategy-research](https://github.com/aaronpascalkujur/trading-strategy-research) | 7 strategie contro l'aritmetica dei costi | Il carry è l'unico sopravvissuto; 11.57%/anno sul notional → 4.39% dopo tasse → **perde contro un titolo di stato al 5.63%** |
| [zwmjj/funding-rate-arb](https://github.com/zwmjj/funding-rate-arb) | BTC/ETH Binance, 6 anni | 9.0%/11.4% lordo, 18/14 trade, 100% win rate (che loro stessi trattano come avvertimento). **79-82% del PnL da trade pre-2022; zero trade nel 2025-2026** |
| [nsheng1568/funding-dispersion-trade](https://github.com/nsheng1568/funding-dispersion-trade) | Trasversale long-short su HL | ~7% APR netto a ~25% vol, Sharpe ~0.28. Convergenza indipendente su EWMA 168h |

**Riconciliazione del divario 1% vs 9-12%:** (a) loro quotano sul notional, io sul
capitale (2×); (b) nessuno modella basis o liquidazioni; (c) **il mio campione
cade interamente nel periodo in cui il loro backtest non apre posizioni.**

Accademia: Koijen, Moskowitz, Pedersen & Vrugt (2018) formalizzano il carry su
azioni, bond, valute e commodity 1972-2012, positivo in tutte e quattro e
**largamente decorrelato fra classi**. Un'analisi su 26 exchange trova che gli
spread di funding mostrano "persistenza estremamente elevata".

⚠️ Lo SSRN *"Failure of Cross-Sectional Alpha Screening on Cryptocurrency
Perpetual Futures"* **sembra** una smentita e non lo è: misura l'IC di
*funding → rendimento di prezzo* su panieri long-short direzionali. Qui si misura
*funding → funding* su posizioni delta-neutral per coin.

---

## 11-bis. Stagionalità delle commodity — indagata, per lo più negativa

Domanda posta: si può usare la stagionalità come segnale invece di subirla come
contaminante? Indagata senza scrivere codice; ecco cosa è emerso.

**Due stagionalità diverse, e solo una paga.**
- *Stagionalità-previsione* (il gas sale d'inverno): pubblicamente nota,
  meccanicamente prevedibile, e **già dentro la curva**. La forma della curva
  forward *è* la previsione stagionale del mercato. Scommetterci significa
  scommettere che la curva sottoprezzi un fatto scritto in ogni manuale.
- *Stagionalità-premio al rischio*: compenso per aver assorbito un rischio
  specifico e stagionale (es. rischio meteo intorno al raccolto, che gli hedger
  pagano per scaricare). Questa può pagare, perché qualcuno *deve* pagarla — la
  stessa struttura del funding.

**Lo spazio di ricerca di uno scanner stagionale** (~50 commodity, spread
calendario e inter-commodity, finestre entrata/uscita): **~120 milioni di
combinazioni**. Con 30 anni di storico, per puro caso ci si attende:

| anni vincenti | combinazioni attese per caso |
|---|---|
| 24/30 (80%) | **86.346** |
| 27/30 (90%) | **509** |
| 28/30 (93%) | 52 |
| 29/30 (97%) | 3 |

"Ha vinto 27 anni su 30" è ciò che si trova centinaia di volte in dati casuali.

**Evidenza out-of-sample** (arXiv 2609.12227, 2026): 15 commodity liquide,
2016-2024, finestre rolling di 10 anni, tre metodi (DVR, SSA, RLSSA), **con costi
e correzione per confronti multipli**. Nessun modello stagionale batte un
benchmark equal-weight long (Sharpe 0.191). Il migliore ha rendimenti cumulati
mediani negativi.
⚠️ Testavano posizioni *outright* mensili, **non spread calendario**. Non è quindi
una confutazione completa della variante a spread.

**Dove l'idea regge: il vincolo di full carry.** Lo spread fra due mesi adiacenti
non può superare il costo di stoccaggio + finanziamento + assicurazione in
contango, perché altrimenti si compra il vicino, si stocca e si vende il lontano.
In backwardation **non esiste un vincolo simmetrico**. Questa asimmetria è
meccanica e fisica, non statistica — è la versione più solida dell'idea e l'unica
che meriti un test.

**SeasonAlgo**: SaaS, 30 anni di storico, nessuna API o export documentati.
Descrive sé stesso come "backtesting e ottimizzazione di qualsiasi strategia
stagionale su tutto lo storico", che è precisamente la ricerca quantificata sopra.
**Non serve**: la stagionalità si calcola meglio in proprio (leave-one-out), e
quello che pubblica sono statistiche derivate, non curve grezze.

### Misurato: la stagionalità NASCONDE il carry, non lo fornisce

Dati EIA (curve energia, contratti 1-4, gratuiti e ufficiali — vedi §9-bis).
Autocorrelazione di rango del carry mensile, grezzo contro destagionalizzato
leave-one-out:

| commodity | R² stagionale | 3m grezzo | 3m residuo | 6m grezzo | 6m residuo | 12m grezzo | 12m residuo |
|---|---|---|---|---|---|---|---|
| WTI | −4.3% | 0.617 | 0.605 | 0.442 | 0.442 | 0.157 | 0.157 |
| **NATGAS** | 35.7% | **−0.102** | **+0.338** | 0.305 | 0.156 | 0.502 | **0.086** |
| HEATOIL | 9.4% | 0.578 | 0.603 | 0.240 | 0.326 | 0.349 | 0.233 |
| **GASOLINE** | **71.7%** | 0.094 | **0.306** | −0.564 | −0.024 | 0.699 | **0.257** |

Due letture opposte a seconda dell'orizzonte:

- **A 12 mesi la persistenza apparente È stagionalità.** NATGAS crolla da 0.502 a
  0.086, GASOLINE da 0.699 a 0.257. Dodici mesi di distanza è lo stesso mese di
  calendario: stai "prevedendo" che novembre somigli ai novembre passati.
- **A 1-3 mesi — l'orizzonte tradabile — la persistenza NON è stagionale, e
  destagionalizzare la rafforza.** NATGAS passa da **−0.102 a +0.338** a 3 mesi,
  GASOLINE da 0.094 a 0.306. Il ciclo stagionale cambia segno in quell'arco e
  *maschera* il segnale sottostante.

**Conclusione controintuitiva:** la stagionalità non è l'edge, è il rumore che
nasconde l'edge. Non si trada — si sottrae per vedere il carry sotto. È l'inverso
esatto di quello che fa uno scanner stagionale.

⚠️ Cautele: sono autocorrelazioni *temporali* su 4 commodity, non l'IC
*trasversale* misurato sul cripto (0.65) — grandezze diverse, non confrontabili
direttamente. Il lag a 1 mese (~0.86) è in parte meccanico su una serie lenta
mediata mensilmente: le colonne informative sono 3m e 6m. Dati fermi al 2024-04.

Implementato in `seasonality.py` con 9 test. Bug trovato dai test:
`seasonality_explains` = `1 − res/raw` è mal definito con `raw` negativo (dava
+2.53 dove la verità era l'opposto); ora restituisce `None` e c'è `masks_signal`
per quel caso.

---

## 12. Cosa manca, in ordine di valore

1. **Ribilanciamento della copertura.** L'unica leva vista spostare il risultato di
   un ordine di grandezza (basis da −$1.322 a −$70). Ma la letteratura avverte:
   una posizione da $100k ribilanciata 3×/settimana a 4bp costa ~$600/mese contro
   ~$900/mese di funding lordo. **Il parametro da ottimizzare è la soglia di
   deriva, non la frequenza.**
2. **Il run a 540g usa ancora slippage piatto 5bp.** Il costo per-coin e quello
   asimmetrico (ingresso mediana / uscita p99) sono costruiti e testati ma **non
   applicati** al backtest MEXC. Quindi +0.96% è probabilmente ancora ottimista.
3. **Walk-forward.** Le 144 configurazioni sono in-sample.
4. **Latenza di esecuzione** fra il fill della prima gamba e della seconda: l'unica
   variabile che né i book pubblici né gli archivi possono dare.
5. **Estensione a commodity/ETF** — vedi `NEXT_SESSION.md`.
