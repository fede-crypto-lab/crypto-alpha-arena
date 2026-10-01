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
misurato su 540 giorni con basis reale e universo liquido. **È un quarto del
risk-free** (4.14%): vedi §1-bis per la tabella completa in euro e le soglie di
capitale. Ogni volta che ho sostituito un'ipotesi con una misura, la stima è scesa.

---

## 1-bis. Le percentuali, e il confronto che decide tutto

| strategia | APR sul capitale | 10k € | 50k € | 100k € | vs risk-free EUR (50k) |
|---|---|---|---|---|---|
| Carry cripto, **regime magro** — *misurato, 540g* | **0.96%** | €96 | €480 | €960 | **−769 €** |
| Carry cripto, regime storico — *stima 2×, non misurata* | 2.00% | €200 | €1.000 | €2.000 | −250 € |
| Carry commodity, ipotesi prudente — **non misurata** | 4.00% | €400 | €2.000 | €4.000 | +750 € |
| Carry commodity, ipotesi Koijen (Sharpe 0.7) — **non misurata** | 7.00% | €700 | €3.500 | €7.000 | +2.250 € |
| **Risk-free EUR — BCE deposit facility (23-09-2026)** | **2.50%** | €250 | €1.250 | €2.500 | — |

### Quale risk-free: EUR, non USD

Una versione precedente di questa tabella usava il T-bill USA a 3 mesi (4.17%).
**È il metro sbagliato per un investitore in euro**: comprare Treasury in dollari
aggiunge rischio cambio, e coprirlo costa all'incirca il differenziale di tasso
(parità coperta dei tassi), riportando il rendimento a quello in euro. Il
riferimento corretto è il tasso euro:

| tasso | valore | uso |
|---|---|---|
| BCE deposit facility (`ECBDFR`) | **2.50%** | **il metro** — quanto rende la liquidità in euro |
| T-bill 3m USA (`DGS3MO`) | 4.17% | in dollari, non confrontabile senza copertura |
| BTP 10 anni (`IRLTLT01ITM156N`) | 3.99% | ha rischio tasso e duration: non è risk-free |

La correzione cambia la conclusione di grado ma non di segno: il carry misurato
perde contro il risk-free di **2.6×** invece che di 4.3×, e il regime storico
stimato (2.00%) arriverebbe **quasi alla pari** con il 2.50%.

⚠️ I tassi si muovono. Riprendi `ECBDFR` da FRED
(`https://fred.stlouisfed.org/graph/fredgraph.csv?id=ECBDFR`, raggiungibile)
prima di fidarti della colonna di destra.

### Da cosa nasce lo 0.96% — la catena, e su che base

La base è **100.000 $ di capitale**, che sostengono 50.000 $ di long spot più
50.000 $ di short perp. Non è il rendimento sul notional: è sul capitale che va
parcheggiato e non è impiegabile altrove.

| passaggio | risultato |
|---|---|
| Funding lordo $2.459 su notional medio in posizione di $28.925 | **5.75% APR** ← *il numero che pubblicizzano le guide* |
| Book dimensionato per 5 slot = $50.000 di capacità, utilizzo **57.9%**: il capitale fermo resta nel denominatore | 3.33% APR |
| A leva 1× ogni slot immobilizza **2× notional** (spot intero + margine perp) | 1.66% APR |
| Meno basis (−$70) e commissioni (−$966 = **39% del funding lordo**) | **0.96% APR** |

I tre punti di perdita, in ordine: **utilizzo** (42% del tempo fermo),
**struttura del capitale** (2× notional a leva 1×), **commissioni**.

### Monitoraggio: due frequenze, e la soglia che fa scattare l'azione

Le due grandezze si muovono a velocità diverse e non vanno misurate insieme.

- **Livello del funding — settimanale.** Poche chiamate API.
  `--persistence --days 180 --universe-size 40`
- **Struttura del segnale — trimestrale.** ~20 minuti e molto download; la
  persistenza si muove lentissima (ρ 0.65 sia a 180g sia a 3 anni).
  `--persistence --days 1095`

**Soglia d'azione:** oggi il top quintile rende ~12% APR e la strategia netta
0.96%. Perché superi il 2.50% euro serve un funding intorno all'**11% sul notional
impiegato**, cioè un top quintile sopra il **20-25% APR** — dove stava in media
negli ultimi 3 anni (26.7%). **Allerta sopra il 20% APR sostenuto.**

### La strategia che funziona, operativamente

Carry trasversale delta-neutral, la configurazione da cui esce lo 0.96%:

> 5 slot di capitale. Ogni 8 ore classifichi ~20 coin per funding realizzato nelle
> ultime 168h. Apri i primi 5 comprando spot e vendendo il perp sullo stesso
> notional, **leva 1× sulla gamba perp**. Esci solo quando una coin scende sotto il
> 20° posto **e** sono passati almeno 20 giorni. Escludi le coin con p99 di
> slippage sopra i 40bp e il quintile più volatile.

Hold medio 65 giorni, 16 rotazioni l'anno, win rate 75% [CI 55-88%], 3
liquidazioni su 24 rotazioni, max drawdown 2.73%.

### Soglie di capitale

- **10-50k**: il carry cripto non ha senso economico contro il risk-free EUR. Non
  è un problema di taratura: il divario è di 2.6×, non di qualche punto base.
- **100k+**: il carry commodity *potrebbe* averlo, ma solo avvicinandosi
  all'ipotesi alta — che è la stima di Koijen su scala istituzionale, 1972-2012,
  con contratti full-size e non micro.

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
| IBKR/TWS | 🔑 | Universo ampio, ma **scaduti solo fino a 2 anni dopo la scadenza** (doc IBKR, `includeExpired`): curva viva + ~2-3 anni. Basta per la persistenza del carry, **non** per il walk-forward stagionale a 15 anni. Solo via TWS/IB Gateway, non app mobile/web |
| Databento `GLBX.MDP3` | 💳 | CME Globex **dal 6 giugno 2010**, tutte le scadenze, OHLCV giornaliero; a consumo, **125 $ di credito gratuito** ai nuovi account. `hist.databento.com` **raggiungibile dal container** (401 senza chiave). Pacchetto `databento` su PyPI. Scaricatore: `fetch_databento.py` (stima costo gratis, scarica solo con `--confirm`, tetto `--max-usd`) |
| Databento `IFUS.IMPACT` | 💳 | ICE US (cacao, caffè, zucchero, cotone, succo d'arancia) **solo dal 23-12-2018**: troppo corto per il walk-forward stagionale |
| TradingView | ✋ | Contratti scaduti NYMEX disponibili sul piano dell'utente, ma **solo export manuale** (nessuna API ufficiale; gli scraper violano i termini). Usato per il crack RB-CL 2014-2025 |

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

## 11-ter. Commodity: il carry c'è, ma il long-short direzionale non lo cattura

Misurato su dati EIA (4 commodity energia, curve C1-C4). **Correzione importante
rispetto a come avevo posto l'analogia**: il carry commodity NON è
strutturalmente identico al cash-and-carry cripto.

- **Cripto**: spot contro perp dello *stesso* asset → rischio prezzo ≈ 0 per
  costruzione.
- **Commodity, roll yield**: si incassa con una posizione **direzionale** sul
  front month; la neutralità si ottiene solo lunghi su alto-carry e corti su
  basso-carry, cioè un long-short di paniere **con rischio di prezzo relativo**.
- **Commodity, spread calendario**: delta-neutral davvero, ma il motore è la
  convergenza e l'economia dello stoccaggio, non il roll yield. **Non testato.**

### La materia prima c'è, e in abbondanza

Quanto spesso il carry supera il breakeven dei costi (2.18% a 14g su MCL):

| commodity | giorni | media | % in backwardation | % sopra soglia | carry medio quando sopra |
|---|---|---|---|---|---|
| WTI | 9.492 | +0.3% | 46% | **42%** | **+21.6% APR** |
| NATGAS | 7.407 | **−24.2%** | 19% | 17% | +43.4% APR |
| HEATOIL | 6.044 | +0.4% | 35% | 30% | +26.3% APR |
| GASOLINE | 4.582 | +4.1% | 61% | **56%** | **+28.0% APR** |

Rapporto segnale/soglia **10-20×**, contro **0.6×** del cripto (edge ~5% APR
contro breakeven 8.86%). Il gas è in contango strutturale (−24% medio): è il
costo dello stoccaggio, non un'anomalia.

### Ma il segnale non diventa rendimento

Long-short top/bottom carry, ribilanciato mensilmente, 222 mesi (2005-2024):

| | valore |
|---|---|
| segnale (spread di carry) | **63.6% APR** |
| **rendimento realizzato lordo** | **5.7% APR** |
| volatilità realizzata | **42.6%** |
| **Sharpe** | **0.13** (0.11 netto costi) |
| mesi positivi | 55% |
| peggior mese | **−48.5%** |

**Il carry si è tradotto in rendimento solo per il 9% della sua grandezza.** Quel
42.6% di volatilità è l'intera differenza col cripto, dove il book delta-neutral
aveva volatilità ≈ 0 per costruzione.

I costi sono irrilevanti qui: 1.01% APR su 24 gambe MCL l'anno, contro il 39% del
lordo che si mangiavano nel cripto. **Nelle commodity il problema non sono i
costi, è il rischio di prezzo** — l'esatto opposto del cripto.

### Due errori miei in questa analisi, corretti

1. **"100% dei mesi con spread positivo"** è una tautologia: il massimo di quattro
   numeri è sempre sopra il minimo. Non è un win rate.
2. Il 44% dei mesi la coppia era *lunga benzina / corta gas*. Stare corti sul gas
   non è raccogliere carry: è una direzionale sulla commodity più volatile che
   esista, e quel −24% è il compenso per quel rischio.

### Perché il test è informativo ma sul peggior universo possibile

4 commodity, **tutte energia**, 2 posizioni per volta. Koijen riporta Sharpe ~0.7
su 20+ commodity e più settori. Proiettando la sola diversificazione (ipotesi:
75% del rischio idiosincratico, **non verificata**):

| posizioni | vol attesa | Sharpe |
|---|---|---|
| 2 (misurato) | 42.6% | **0.13** |
| 10 | 26.9% | 0.21 |
| 20 | 24.3% | 0.23 |

**La diversificazione da sola non chiude il divario con 0.7.** Le spiegazioni
possibili — universo mono-settore, correzione del roll troppo grezza, erosione del
fattore, periodo — non sono distinguibili senza dati più ampi. Non trattare lo 0.23
come una previsione: la quota idiosincratica è un'assunzione, e con più commodity
anche lo *spread* di carry si restringerebbe (le 4 energy hanno uno spread estremo
perché il gas è un outlier).

### Cosa ne consegue per il piano

- **Il long-short energy-only è morto.** Misurato, Sharpe 0.13, peggior mese −48.5%.
- **La breadth non è opzionale**: servono 12-20 commodity su più settori, ed è
  esattamente ciò che il pull IBKR deve dare.
- **Lo spread calendario resta il ramo più interessante**, perché è l'unica
  struttura che conserva ciò che rendeva attraente il carry cripto — la
  delta-neutralità vera. È **non testato**.
- **Reset delle aspettative**: il carry commodity è un fattore **direzionale e
  volatile**, non una rendita sicura. Anche funzionando bene, è Sharpe 0.4-0.7 con
  volatilità a due cifre — un profilo di rischio completamente diverso da quello
  cercato all'inizio.

---

## 11-quater. Spread stagionali: il claim "sempre vincenti" testato walk-forward

Domanda: SeasonAlgo mostra spread che vincono da anni — continuano a vincere?
Testato con `seasonal_walkforward.py`, che riproduce la selezione di uno scanner
(~1.000 finestre entrata/durata/direzione per spread, tieni quelle con ≥12/15
anni vincenti) e poi **guarda l'anno successivo**, che nessuno scanner mostra.

Dati: prezzi **spot** EIA (nessuna rollata), 1986-2026, costo 0.05 $/bbl a trade.

| spread | trovati/anno | per caso | IS win | **OOS win** | baseline | best pick OOS [CI 95%] |
|---|---|---|---|---|---|---|
| **crack benzina (RB-CL)** | 34.1 | 18.0 | 83% | **64%** | 50% | **23/26 [71-96%]** |
| crack gasolio (HO-CL) | 16.3 | 18.0 | 82% | 56% | 49% | 18/26 [50-83%] |
| **benzina vs gasolio** | 38.6 | 18.0 | 84% | **69%** | 49% | 20/26 [58-89%] |
| Brent − WTI | 9.1 | 18.0 | 81% | 51% | 49% | 13/25 [33-70%] |

Alzando la soglia ("sempre vincenti"):

| spread | soglia | per caso/anno | trovati/anno | IS | OOS |
|---|---|---|---|---|---|
| crack benzina | 14/15 | 0.50 | 3.0 (**6×**) | 94% | **76%** |
| crack benzina | 15/15 | 0.03 | 0.3 | 100% | 78% |
| benzina vs gasolio | 14/15 | 0.50 | 5.5 (**11×**) | 95% | **75%** |
| crack gasolio | 14/15 | 0.50 | 0.7 (≈ caso) | 94% | 59% |

### Tre letture

1. **"Sempre vincente" non vuol dire "vincerà".** Anche sugli spread migliori il
   100% storico diventa **~75% l'anno dopo**. Lo scanner sovrastima di 20-25 punti.
2. **La diagnosi pratica è "trovati contro per caso".** Se lo scanner trova circa
   quanti pattern produrrebbe il caso (crack gasolio 16 vs 18; Brent-WTI 9 vs 18),
   sono rumore e falliscono fuori campione. Se ne trova molti di più (crack benzina
   6×, benzina vs gasolio 11× a 14/15) c'è stagionalità vera, e regge.
3. **La stabilità della scelta separa il vero dal rumore.** Sul crack benzina il
   processo sceglie **la stessa finestra in 25 anni su 26**: *long dal 21-26
   gennaio per 90 giorni*. Meccanismo fisico: manutenzione delle raffinerie a
   feb-mar, passaggio alla benzina estiva (più costosa da produrre), scorte che
   calano prima della driving season. Su benzina vs gasolio invece la finestra
   salta fra "long febbraio" e "short settembre": più rumorosa.

### Il crack benzina, anno per anno (best pick, fuori campione)

**23/26 vinti. Totale +192 $/bbl = +192.490 $ per spread da 1.000 barili in 26
anni, media +7.403 $/anno, peggior anno −1.758 $.** Le tre perdite sono piccole
(−286, −1.758, −1.732) rispetto alle vincite. È la **prima strategia di tutto il
progetto** con limite inferiore di Wilson ben sopra il 50% su osservazioni
indipendenti fuori campione (71%).

Benzina vs gasolio: 20/26, +126.590 $ totali, media +4.869 $/anno, ma peggior anno
**−11.348 $** e il 2026 in perdita (−9.668 $). Più volatile.

### ⚠️ Il caveat che può cambiare tutto: spot ≠ futures

Il test è sui prezzi **spot**. Ma si tradano i **futures**, e il futures di maggio
sulla benzina a gennaio **incorpora già** l'aspettativa della primavera. Chi compra
il crack sui futures non incassa la stagionalità spot: incassa solo la parte che la
curva futures **non aveva previsto**. La stagionalità fisica è reale e forte (questo
è dimostrato); **quanta ne resta catturabile sui futures non è misurato** — ed è
l'unica domanda che conta per il trading.

Altri limiti:
- **2020 da solo vale +48.812 $** (WTI spot negativo ad aprile, crack esploso).
  Senza il 2020 il totale resta +143.678 $, ma sui futures quell'anno sarebbe stato
  molto diverso.
- Le finestre selezionate si sovrappongono: gli 886 trade "selezionati" non sono
  indipendenti. Il campione onesto è il best pick, **una osservazione per anno**.
- Code enormi negli spread selezionati: peggior singolo trade −29.73 $/bbl (crack
  benzina), −65.86 $/bbl (benzina vs gasolio, crisi distillati 2022).
- **Taglia minima**: un crack RB-CL è 1.000 barili (~100.000 $ di notional per
  gamba). Per quanto ne so **non esiste un micro RBOB**: da verificare con IBKR, ed è
  un vincolo reale per capitali piccoli.

### Cosa fa la prossima sessione con questo

Rigira `walk_forward` **sui contratti futures reali** (dati IBKR), sugli spread che
SeasonAlgo mostra davvero (es. RB maggio − CL maggio). Il modulo è generico: gli
basta un `{data: valore_dello_spread}`. Se l'OOS win e la stabilità della finestra
reggono anche lì, c'è una strategia. Se crollano, la stagionalità era già nel prezzo.

---

## 11-quinquies. SeasonAlgo (demo, solo mais) — lettura della schermata

Dati dalla schermata dell'utente (Strategies → Search, ingresso settembre 2026,
History 15 anni, finestre 3 e 6 mesi): **45 righe, tutte spread calendario sul
mais (ZC)**, win 80-100%. Il demo gratuito copre solo il mais.

- **45 righe ≠ 45 opportunità.** Sono ~5 trade distinti (Z26-H27, Z26-K27,
  Z26-N27, H27-K27, butterfly H27/K27/N27 e Z26/N27/Z27) ripetuti con date di
  ingresso/uscita spostate di pochi giorni. Il butterfly H27-2K27+N27 compare 8 volte.
- **Win% con 15 anni, e cosa darebbe il caso** (moneta 50/50, prima di ogni
  correlazione fra combinazioni):

  | anni vinti | Wilson 95% | P(≥k) a caso | attesi a caso ogni 100k combinazioni |
  |---|---|---|---|
  | 12/15 (80%) | 55-93% | 1.76% | ~1.760 |
  | 13/15 (87%) | 62-96% | 0.37% | ~370 |
  | 14/15 (93%) | 70-99% | 0.049% | ~49 |
  | 15/15 (100%) | 80-100% | 0.003% | ~3 |

  Uno scanner su spread × giorno d'ingresso × durata supera facilmente 100k
  combinazioni: una lista così si ottiene **anche senza edge**. Il Win% mostrato è
  in-sample; il test che conta è quello di §11-quater (scegli coi 15 anni
  precedenti, verifica sull'anno dopo). Sugli spot EIA la caduta è stata 83% → 64%.
- **Costi (stima, da verificare sul fill):** 1 tick ZC = 0,25¢ = $12,50;
  IBKR ~$2,5 per contratto per lato. Spread a 2 gambe ~$22 a giro → mangia
  11-15% del PLØ su Z26-H27/K27/N27 (150-207 $), **45% su H27-K27** (50 $).
  Butterfly a 4 contratti ~$45 → **fino al 65%** del profitto medio: scartare.
- **C'è un motivo strutturale, non solo statistico.** I trade migliori sono
  *bull spread* sul mais (long scadenza vicina, short lontana). In una commodity
  stoccabile lo spread non può andare oltre il **full carry** (stoccaggio +
  interessi): la perdita è limitata, il guadagno no. Coerente con WorstØ piccolo
  (−42/−78 $) contro BestØ 220-320 $ e RRR 2,5-5. È la stessa asimmetria di §11-bis.
- **Capitale:** il margine di uno spread calendario sul mais è una frazione di
  quello del contratto secco (credito intra-commodity CME). È la strada più
  compatibile con capitali piccoli fra quelle viste. Cifra esatta da leggere su
  IBKR, non stimata qui.

### Stagione 2026 in corso — tracciamento paper (nessun ordine)

Dai grafici SeasonAlgo (linea nera = 2026; ingressi letti dal grafico, ±0,25¢;
"ultimo" = valore LAST in legenda, ~25 settembre 2026). 1¢ = 50 $.

| spread | ingresso | livello ingresso | ultimo | già fatto | PLØ storico | quota del PLØ già incassata |
|---|---|---|---|---|---|---|
| Z26-H27 | 05-09 | ≈ −15,5 | −13,75 | +1,75¢ = +88 $ | 150 $ (3,0¢) | ~58% |
| Z26-K27 | 08-09 | ≈ −23,0 | −20,50 | +2,50¢ = +125 $ | 207 $ (4,15¢) | ~60% |
| Z26-N27 | 05-09 | ≈ −25,25 | −23,25 | +2,00¢ = +100 $ | 177 $ (3,55¢) | ~56% |

- I tre sono **la stessa scommessa** (Dicembre forte contro una scadenza
  differita): nessuna diversificazione fra loro.
- Pattern 5 e 15 anni quasi sovrapposti da settembre a novembre (bene: il
  comportamento non dipende solo dagli anni vecchi). Forma comune: lo spread si
  indebolisce dalla primavera al minimo del raccolto (fine agosto/inizio
  settembre), poi recupera verso dicembre.
- Entrare ora significa partire più lontano dal full carry: meno protezione
  sul ribasso e metà del movimento medio già passata. Per il 2026 si osserva,
  non si entra. Verifica all'uscita (16-21 novembre).
- I grafici mostrano **medie**: non dicono quali anni hanno perso né di quanto.

### Dic/Lug (ZCZ-ZCN, long 5 set → 16 nov): 72 anni — il vantaggio c'è solo negli anni usati per sceglierlo

Tabella anno per anno da SeasonAlgo (`multi-analyze/backtest/ZCZ26-ZCN27/BUY/2026-09-05/2026-11-16`),
analizzata con `seasonal_holdout.py`, costo 22,5 $ a giro. "Vinto" = positivo
**dopo** i costi (SeasonAlgo conta come vinto anche il 2012 a 0 $ e il 2014/2018 a +12,5 $).

| campione | anni | vinti netti | Wilson 95% | medio lordo | medio netto |
|---|---|---|---|---|---|
| selezione SeasonAlgo 2011-2025 | 15 | 10/15 (67%) | 42-85% | +178 $ | +155 $ |
| 15 anni prima, 1996-2010 (mai visti dallo scanner) | 15 | 4/15 (27%) | 11-52% | −43 $ | −66 $ |
| tutti i 57 anni prima, 1954-2010 | 57 | 23/57 (40%) | 29-53% | −18 $ | −41 $ |
| tutti i 72 anni | 72 | — | — | +22 $ | **≈ 0 $** |

- Per decennio (netti): '50 3/6, '60 5/10, '70 3/10, '80 6/10, '90 4/10,
  2000 2/10, 2010 5/10, 2020 5/6. Il grafico cumulativo di SeasonAlgo è piatto o
  in calo dal 1954 al ~2011 e sale solo dopo.
- **In nessun anno** i 15 anni precedenti mostravano ≥80% di vittorie nette:
  questa finestra non sarebbe mai stata selezionabile prima di oggi.
- Due letture: (a) selezione — lo scanner ha trovato le date che calzano gli
  ultimi 15 anni; (b) cambio di regime dopo il 2010. Il test non le distingue;
  la (a) è quella attesa di default (§11-quater: spot EIA 83% → 64%).
- Profondità del contango all'ingresso vs profitto: correlazione −0,23 dal 1985
  (debole; più contango = un po' meglio, come vuole la teoria del full carry).
- **Il test "holdout all'indietro" si fa gratis su SeasonAlgo**: selezione sugli
  ultimi 15 anni, verifica sui 15 prima. Criterio proposto: holdout con vittorie
  nette ≥ 60% **e** medio netto > 0, altrimenti scartare.

### Dic/Mag (ZCZ-ZCK, long 8 set → 21 nov): 79 anni — stesso verdetto, un po' meno netto

| campione | anni | vinti netti | Wilson 95% | medio lordo | medio netto |
|---|---|---|---|---|---|
| selezione 2011-2025 | 15 | 12/15 (80%) | 55-93% | +208 $ | +185 $ |
| 1996-2010 (holdout) | 15 | 6/15 (40%) | 20-64% | +16 $ | −7 $ |
| 1947-2010 (holdout) | 64 | 26/64 (41%) | 29-53% | +13 $ | −10 $ |

Decenni (netti): '40 1/3, '50 5/10, '60 4/10, '70 2/10, '80 6/10, '90 4/10,
2000 4/10, 2010 7/10, 2020 5/6. Mai selezionabile ex ante (0 anni). **Scartato**
col criterio holdout ≥60% e netto > 0. Dic/Lug e Dic/Mag sono la stessa scommessa:
due conferme non indipendenti dello stesso fatto, cioè che la scommessa "Dicembre
forte dopo il raccolto" ha pagato dal 2011 e non prima.

### Dic/Mar (ZCZ-ZCH, long 5 set → 19 nov): 77 anni — terzo scarto

Selezione 2011-2025: 12/15 netti, +128 $ netti. Holdout 1996-2010: **4/15, −12 $**.
Holdout 1949-2010: 24/62 (39%), −13 $. Mai selezionabile ex ante.
**Tutti e tre gli spread SeasonAlgo sul mais falliscono il filtro holdout.** Vanno
bene e male negli stessi decenni ('80 e post-2010 sì; '70, '90, 2000 no).

### Ipotesi "funziona con scorte abbondanti" (USDA PSD) — testata, non regge

Fonte: USDA FAS PSD, `apps.fas.usda.gov/psdonline/downloads/psd_grains_pulses_csv.zip`
(pubblico, raggiungibile dal container, US corn 1960-2026). Variabile ex ante per
l'ingresso di settembre dell'anno Y: scorte iniziali MY Y / consumo+export MY Y−1.
Ipotesi fissata prima di guardare: più scorte → spread Dicembre più forte.

| spread | Spearman holdout 1961-2010 (n=50) | Spearman 1961-2025 | regola "S/U > mediana dei 15 anni prima", 1976-2025: ON / OFF |
|---|---|---|---|
| Dic/Mar | +0,26 | −0,02 | 12/20 (+46 $) / 13/30 (+30 $) |
| Dic/Mag | +0,15 | −0,08 | 13/20 (+89 $) / 14/30 (+38 $) |
| Dic/Lug | +0,23 | −0,01 | 11/20 (+70 $) / 12/30 (−11 $) |

Segno giusto nell'holdout ma **non significativo** (serve ρ ≳ 0,28 con n=50), e
dopo il 2010 gli anni OFF sono andati *meglio* degli ON: le scorte non spiegano il
cambio di regime. Wilson inferiore degli anni ON 34-43%, sotto il 55% richiesto.
Con le scorte finali (che leggono il futuro) va peggio, non meglio. 2026 sarebbe ON
(0,115 contro mediana 0,103), per quel che vale.

### Search con History 30 anni (ingresso settembre 2026): filone mais CHIUSO

6 righe, **2 operazioni distinte, entrambe butterfly** (4 contratti); nessuno
spread a 2 gambe sopravvive a 30 anni (coerente con l'holdout sopra):

| operazione | win 30a | PLØ | WorstØ | RRR | netto dopo ~32-45 $ di costi |
|---|---|---|---|---|---|
| BUY H27−2·K27+N27, ingresso 8-16 set, uscita fine dic/inizio gen | 87-90% | 50-54 $ | −54/−65 $ | 1,9-2,3 | **5-20 $** |
| SELL N27−2·U27+Z27 (vecchio/nuovo raccolto), ingresso 12 set | 80% | 67-79 $ | −248/−254 $ | 1,1 | 22-47 $, coda −250 $ |

Anche se il 90% fosse reale, una volta l'anno per 5-20 $ netti non vale il rischio
di esecuzione (1 tick = 12,5 $ = un quarto del lordo). Non serve l'holdout: il
verdetto non dipende da lui. **Spread stagionali sul mais: nessun candidato.**

Ultimo controllo, Mar/Mag (ZCH-ZCK, long 18 set → 25 ott, 77 anni): selezione
10/15 netti, +41 $; holdout 1996-2010 **5/15, −31 $**; 1950-2010 17/62 (27%), −29 $.
Anni '60 0/10, '70 1/10. PLØ 63 $ lordi: anche se fosse vero, i costi ne mangiano
un terzo. Scartato.

### Crack benzina sui futures: criterio fissato PRIMA di vedere i dati

Dati: export manuali da TradingView (Export chart data, giornaliero, contratti
scaduti RBM/CLM per anno), letti da `tv_crack.py`. Finestra fissa: long RB×42−CL
giugno, 21 gennaio + 90 giorni (scelta sugli spot EIA, quindi ogni anno futures è
fuori campione). Costo stimato 30 $ a giro. **Passa se:** vittorie nette ≥ 70%
degli anni disponibili, medio netto > 0 anche senza il 2020, e le finestre vicine
(11/01, 01/02; 60 e 90 giorni) con medio netto dello stesso segno. Altrimenti la
stagionalità spot era già nel prezzo dei futures e il filone stagionale è chiuso.

### Crack benzina sui FUTURES (TradingView, RBM/CLM 2014-2025): criterio NON superato

`tv_crack.py` su 24 export TradingView (tutti validati: NYMEX, prezzi plausibili,
gambe allineate giorno per giorno). Long RBM×42 − CLM, 21 gen + 90 giorni, costo
stimato 30 $:

| anno | P&L $ | anno | P&L $ | anno | P&L $ |
|---|---|---|---|---|---|
| 2014 | +1.879 | 2018 | −3.029 | 2022 | **+11.712** |
| 2015 | +6.106 | 2019 | +6.599 | 2023 | −3.941 |
| 2016 | −614 | 2020 | −6.048 | 2024 | +2.655 |
| 2017 | −1.676 | 2021 | +5.158 | 2025 | +494 |

- **Vinti netti 7/12 (58%, Wilson 32-81%)** contro il 70% richiesto → **fallito**.
  Sugli spot EIA la stessa finestra vinceva 23/26 (88%): la gran parte della
  stagionalità è già nel prezzo dei futures, come suggeriva il salto
  marzo→aprile della curva RB (+0,22 $/gal ≈ +9 $/bbl, benzina estiva).
- Medio netto +1.578 $/anno (senza 2020 +2.274 $), t = 1,07: **non distinguibile
  da zero**. Mediana +1.186 $. Senza il 2022 (crisi raffinazione, +11.712 $) il
  medio lordo scende a +689 $.
- Finestre vicine: medio netto positivo in tutte e 6 (da +185 a +2.384 $), vinti
  6-8/12. Segno stabile, ampiezza piccola rispetto alla dispersione (peggior anno
  −6.048 $ su un contratto da 1.000 barili).
- Estensione possibile a 2006-2013 e 2026 (18 file in più). Domanda solo
  esplorativa (il medio positivo è reale?), non un recupero del criterio: con
  7/12 servirebbero 8 vittorie su 9 per arrivare al 70%.

**Conclusione: la pista stagionale è chiusa col criterio fissato prima.**

(Storico) pista stagionale che era aperta: il crack benzina (§11-quater, OOS 64% su spot
EIA), da verificare sui futures RB−CL con lo stesso holdout (tabella anno per anno
da SeasonAlgo con accesso completo, o Databento).

(Fatto: tabelle anno per anno dei 3 spread a 2 gambe, vedi sopra. La prova "a
ritroso" col Range nel passato è superata dal test holdout.)

---

## 11-sexies. Scan di tutti gli spread CME: criterio fissato PRIMA dei dati

`seasonal_scan.py` su dati Databento (`fetch_databento.py`): 21 prodotti CME
(energia, metalli, grani, bestiame), spread calendario (stesso prodotto, fino a
12 mesi di distanza) e inter-commodity (stesso mese, 1:1 in dollari). ~400
finestre per spread (ingresso 35-330 giorni prima della scadenza, 20-90 giorni),
lookback 8 cicli, qualifica a ≥ 7/8 vittorie nette. Costi stimati: 2,5 $ per
contratto per lato + 1 tick per gamba.

**Costo misurato (stima Databento, 2010-06-06 → 2026-09-30, ohlcv-1d, solo contratti
singoli): 10,05 $ per i 21 prodotti, ~57 MB.** Chiedere per "parent" (`CL.FUT`)
include anche gli spread quotati in borsa: 7× i dati e il costo (CL 7,69 $ contro
1,08 $). Per questo il downloader chiede i simboli singoli (`CLF0`…`CLZ9`).

Domanda unica, globale: **scegliere la finestra sui cicli passati batte il caso
sul ciclo dopo?** Criterio:
- best pick per spread-anno: vittorie nette OOS **≥ baseline + 5 punti** e limite
  inferiore di Wilson sopra la baseline, **e** medio netto > 0;
- **in almeno 6 degli ~8 anni di test** i best pick battono la baseline (gli
  spread dello stesso anno sono correlati: il conteggio per anno è l'unico onesto).

Se fallisce: la stagionalità degli spread CME 2010-2026 è già nei prezzi, filone
chiuso definitivamente. Se passa: secondo stadio sui singoli spread, ricontrollo
sui prezzi di settlement, poi paper trading. **Nessuna lista di "spread vincenti"
va letta prima di questo verdetto.**

### 11-sexies — RISULTATO (dati Databento scaricati 30-09-2026, ~10,20 $ di credito)

21 prodotti, 2.915 contratti, 2010-06 → 2026-09. Pulizia necessaria e testata:
simboli a una cifra riusati dopo 10 anni (CLM9 = giugno 2019 e poi giugno 2029:
2 scambi spuri nella serie 2019), NG passato a simboli a due cifre da maggio 2025
(senza, la storia NG finiva lì in silenzio), contratti ancora vivi esclusi,
serie che non finiscono vicino al mese di consegna escluse.

**Criterio globale (fissato prima): FALLITO.**

| universo | spread | baseline | best pick OOS | medio netto | anni > baseline |
|---|---|---|---|---|---|
| tutti | 3.060 | 48,7% | 51,0% [50,2-51,7] (+2,3 pt) | **−77 $** | 4/9 |
| calendario (stesso prodotto) | 1.587 | 47,7% | **53,9%** [52,8-54,9] (+6,2 pt) | **+194 $** | **7/9** |
| inter-commodity (1:1 in $) | 1.473 | 49,7% | 48,2% (−1,5 pt) | −342 $ | 3/9 |

La divisione calendario / inter-commodity è **post-hoc** (non era nel criterio).
Robustezza del sottogruppo calendario, tutta positiva:
- costi ×2: +8,2 pt, +174 $, 7/9 anni; lookback 6: +5,2/+5,9 pt, 9/11 anni;
  lookback 10: +4,6/+5,2 pt, +198/+291 $, 5/7 anni;
- per settore: energia +5,0 pt (+269 $), metalli +9,0 (+133 $, 9/9 anni),
  grani +5,3 (+86 $), bestiame +7,8 (+183 $).
- **Non è solo carry**: finestra fissa (150 gg prima della scadenza, 90 gg) con
  la sola direzione dal passato → 50,2%, +101 $, mediana 2 $. La scelta stagionale
  aggiunge ~+3,7 pt e ~+90 $.
- Direzione scelta: 2/3 short spread (front debole contro back).

**Il problema è il rischio, non il segno**: per trade media +194 $, deviazione
standard 2.527 $, p1 −6.010 $, **peggiore −32.925 $**, migliore +28.021 $.
Sharpe per trade ~0,08: serve un portafoglio largo per vedere la media, e le
code distruggono un conto piccolo senza dimensionamento e stop.

Prossimo passo onesto (la scoperta è post-hoc, quindi va confermata su dati
MAI visti): stesso test sui calendari 1990-2009 da un'altra fonte (Norgate/CSI),
oppure paper trading in avanti; poi studio di dimensionamento/stop e margini
reali IBKR per spread calendario.

## 11-septies. Filtri d'ingresso tecnici sugli spread calendario — piano fissato PRIMA

Decisione dell'utente (01-10-2026): niente dati pre-2010 (i mercati e le
stagionalità cambiano); migliorare i segnali sui dati che abbiamo.

Universo: i best pick del sottogruppo calendario (§11-sexies: lookback 8, ≥7/8,
~8.400 operazioni OOS 2018-2026). Le caratteristiche si calcolano sullo spread
**solo con prezzi precedenti al giorno d'ingresso**, orientate nella direzione
del trade (positivo = a favore):
1. momentum 20 gg (variazione / volatilità 20 gg);
2. distanza dalla media mobile 20 gg (in volatilità);
3. regime di volatilità: vol 20 gg / mediana vol 120 gg;
4. livello rispetto agli stessi giorni dei cicli precedenti (z-score: "già
   andato" o "ancora da fare");
5. forza del segnale stagionale: t-stat delle 8 vittorie passate;
6. ingresso con conferma: entrare il primo giorno, entro 10 dalla data
   stagionale, in cui il momentum 5 gg gira nella direzione del trade.

**Divisione degli anni:** sviluppo = test year 2018-2022; **cassetto = 2023-2026,
guardato una sola volta** a regola congelata. In sviluppo: terzili per ogni
caratteristica (soglie calcolate solo sullo sviluppo); si sceglie **una** regola
(una caratteristica, o al massimo due combinate) solo se i terzili sono monotoni.

**Criterio sul cassetto:** rispetto ai best pick non filtrati dello stesso
cassetto, medio netto ≥ +25% **e** vittorie ≥ +2 punti **e** meglio in almeno
3 dei 4 anni. Altrimenti: l'analisi tecnica d'ingresso non aggiunge nulla e si
resta sulla regola stagionale semplice.

### 11-septies — sviluppo (2018-2022, 4.790 best pick) e regola CONGELATA

Terzili (soglie solo sullo sviluppo); base non filtrata 51,5%, +59 $, t 1,55:

| caratteristica | basso | medio | alto | monotono |
|---|---|---|---|---|
| momentum 20 gg | 49,9% / −235 $ | 51,1% / +132 $ | 52,6% / +252 $ | sì |
| distanza da MA20 | 49,3% / −213 $ | 50,8% / +134 $ | 53,6% / +228 $ | sì |
| regime di volatilità | 47,8% / −10 $ | 49,5% / +12 $ | 56,4% / +147 $ | sì |
| livello vs anni passati | 51,2% / −116 $ | 53,3% / +22 $ | 50,3% / +275 $ | **no** (vittorie) |
| t-stat stagionale | 49,9% / −11 $ | 51,2% / +23 $ | 53,4% / +166 $ | sì |
| ingresso con conferma 5 gg | 46,7% / −66 $ contro 52,7% / +99 $ degli stessi trade all'ingresso stagionale | | | **peggiora** |

Scelta tra le 4 caratteristiche monotone, da sole o a coppie, per t-stat più
alta: **`vol_regime:high+seasonal_t:high`** (sviluppo n=556, 58,1%, +524 $,
t 4,74). Seconda: `mom20:high` (n=1.462, 52,6%, +252 $, t 4,17). Soglie: vol 20gg
/ vol 120gg ≥ 1,00 e t-stat stagionale ≥ 2,94.
**Congelata qui, prima di guardare 2023-2026.**

### 11-septies — CASSETTO 2023-2026 (guardato una volta): criterio NON superato

| | n | vittorie | medio netto | mediana |
|---|---|---|---|---|
| cassetto, non filtrato | 3.615 | 57,0% | +373 $ | +140 $ |
| cassetto, regola congelata | 279 | 61,6% | +544 $ | +290 $ |

Per anno (non filtrato → filtrato): 2023 57,5%/+788 $ → **72,9%/+1.192 $**;
2024 55,2%/+125 $ → **72,0%/+649 $**; 2025 58,4%/+230 $ → **40,5%/−103 $**;
2026 56,8%/+235 $ → **48,1%/−81 $**.

Criterio: medio +46% ✓, vittorie +4,6 pt ✓, **meglio in 2 anni su 4 ✗** (servivano
3). **Fallito**, e peggiora proprio negli anni più recenti. Il filtro tiene solo
~8% delle operazioni (27-93 per anno): troppo poche per fidarsi del dato aggregato.
La seconda regola (`mom20:high`) NON è stata provata sul cassetto: farlo adesso
sarebbe un secondo tentativo sugli stessi anni e il cassetto non sarebbe più tale.

Conclusione: i filtri tecnici d'ingresso non aggiungono un miglioramento
affidabile; si resta sulla regola stagionale calendario semplice. Nota a margine:
quella regola semplice nel 2023-2026 fa 57,0% e +373 $ (anni già visti in §11-sexies,
quindi non è una conferma indipendente).

## 11-octies. Analisi per settore merceologico (richiesta dall'utente, 01-10-2026)

`sector_report.py`, stesso walk-forward (lookback 8, ≥7/8). Gruppi: metalli
(GC SI HG PL PA), petrolio (CL BZ HO RB), gas naturale (NG), grani e semi oleosi
(ZC ZW KE ZS ZM ZL ZO ZR), bestiame (LE GF HE). Calendario = stesso prodotto;
intra-settore = due prodotti collegati, stesso mese, 1:1 (non neutrale in
nozionale: in parte direzionale). **Descrittivo** (divisione chiesta dopo aver
visto i dati); le colonne ≤2022 / >2022 dicono se regge nelle due metà.

| settore, tipo | spread | baseline | best pick | medio | ≤2022 | >2022 | anni > base | dev. std / peggiore |
|---|---|---|---|---|---|---|---|---|
| **metalli, calendario** | 415 | 46,8% | **55,8%** (W 53,3) | +133 $ | 54,6% / +105 $ | 57,3% / +169 $ | **9/9** | 1.207 / −9.290 $ |
| metalli, intra | 120 | 49,8% | 44,8% | −1.336 $ | | | 2/9 | 21.868 / −159.690 $ |
| petrolio, calendario | 570 | 48,9% | 51,8% (W 50,0) | +115 $ | 51,6% / +11 $ | 52,1% / +250 $ | 5/9 | 3.125 / −32.925 $ |
| petrolio, intra | 72 | 49,4% | 48,2% | −234 $ | | | 6/9 | 4.235 / −25.229 $ |
| gas, calendario | 144 | 47,9% | 59,3% | +753 $ | **49,8% / −93 $** | **75,6% / +2.195 $** | 7/9 | 3.891 / −22.470 $ |
| grani, calendario | 302 | 46,2% | 51,5% | +86 $ | **43,5% / −35 $** | **61,1% / +230 $** | 7/9 | 1.199 / −7.850 $ |
| grani, intra | 138 | 49,4% | 49,4% | −59 $ | 47,0% / −366 $ | 51,5% / +218 $ | 6/9 | 3.580 / −17.810 $ |
| bestiame, calendario | 156 | 48,4% | 56,2% (W 53,1) | +183 $ | **62,1% / +454 $** | **48,1% / −187 $** | 7/9 | 1.927 / −8.680 $ |
| bestiame, intra | 13 | 49,6% | 59,2% (n=103) | +776 $ | 63,6% | 54,2% | 6/9 | 4.530 / −12.598 $ |

Letture:
- **Solo i calendari sui metalli reggono in entrambe le metà e in tutti i 9 anni**,
  con il rischio per trade più contenuto. Per prodotto: oro 61% / +199 $ (n=423),
  rame 56% / +158 $ (n=764); argento 46% / −63 $ e platino 55% / −46 $ negativi.
  Probabile motore: carry finanziario (tassi) più che stagionalità, da verificare.
- Gas e grani: buoni solo dopo il 2022 (regime: crisi gas/GNL, shock grano 2022).
  Bestiame: buono prima, negativo dopo. Petrolio: marginale. → dipendenti dal regime.
- Intra-settore: negativo quasi ovunque; metalli disastroso (nozionali 1:1 molto
  diversi). Eccezioni piccole e da non sopravvalutare: bestiame (n=103), HO-RB
  (64%, n=97), HOQ-RBQ 9/9 anni (lista esplorativa).
- Calendari ricorrenti nei metalli (esplorativo, è una selezione): GCV-GCQ+1 7/8
  anni +1.294 $, HGK-HGF+1 6/7 +1.317 $, HGN-HGF+1 6/7 +1.245 $.

## 11-nonies. Oro e rame: fattori esterni (richiesta utente, 01-10-2026)

Fonti esterne libere: FRED CSV senza chiave (`fredgraph.csv?id=…`), raggiungibile
dal container: DTB3 (BOT 3 mesi), DGS2, DFII10 (reale 10a), DTWEXBGS (dollaro),
SP500, VIXCLS, SOFR. Rame: secondo le fonti il motore degli spread è il livello
delle **scorte** (LME/COMEX/SHFE) — non disponibile liberamente in modo automatico.

Misure:
- **Oro**: carry implicito (ln(F2/F1)/anni sulle prime due scadenze attive) contro
  BOT 3m: correlazione dei livelli **0,95**; premio medio 0,5-1,9 pt (2020: 1,9).
  Variazioni settimanali: 0,08 (rumore delle chiusure UTC).
- **Rame**: carry contro BOT 0,46; dipende da altro (scorte).
- P&L dei best pick calendario contro variazioni dei fattori durante il trade
  (orientate per direzione): oro |r| ≤ 0,19 (il più alto: prezzo dell'oro −0,19,
  meccanico), rame |r| ≤ 0,14. Oro long spread: BOT in calo 73% / +445 $, in salita
  50% / −308 $ (come vuole la teoria), ma vince anche a tassi fermi (58% / +175 $).
  Rame: short spread 60% / +244 $ (n=483), long spread 49% / +12 $ → il vantaggio
  è vendere la scadenza vicina quando è tesa.

### Segnale "carry anomalo" — parametri fissati PRIMA, nessuna ottimizzazione

residuo = carry implicito − BOT 3m; z = (residuo − media 252 gg) / dev.std 252 gg.
z ≤ −1,5 (curva anormalmente piatta/backwardation rispetto ai tassi) → vendi
la scadenza vicina, compra la lontana; z ≥ +1,5 → il contrario. Uscita quando z
torna a 0 o dopo 30 giorni di borsa; una posizione per metallo; coppia di
contratti fissata all'ingresso; costi stimati 2 gambe. Periodi: 2011-2018 e
2019-2026 riportati separati. **Passa se in ENTRAMBI i periodi**: netto totale > 0,
vittorie ≥ 55%, netto totale ≥ 2 × massimo drawdown.

### Carry anomalo — risultato: supera il criterio, ma è quasi certamente un ARTEFATTO

`metals_factors.py`, parametri come fissati sopra, ingresso alla chiusura del
giorno DOPO il segnale:

| | ≤2018 | >2018 |
|---|---|---|
| oro | n=127, 62,2%, +5.440 $, DD 460 $ — PASS | n=90, 61,1%, +9.660 $, DD 980 $ — PASS |
| rame | n=66, 72,7%, +3.978 $, DD 1.242 $ — PASS | n=44, 61,4%, +3.260 $, DD 765 $ — PASS |

Controllo di robustezza (stesso segnale, decisioni ritardate di 1-2 giorni in più):
oro con ingresso a 2 giorni **40,1%, −6.410 $**, 13/16 anni in perdita; a 3 giorni
40,6%, −4.960 $. Rame a 2 giorni 43,6%, +537 $. Con costi ×2 (ingresso a 1 giorno):
oro 51,2%, +8.590 $; rame 56,4%, +3.388 $.

Un'anomalia di carry economica rientra in settimane; qui il vantaggio vive un
giorno → firma delle **chiusure non sincronizzate** (ohlcv-1d chiude a mezzanotte
UTC con l'ultimo scambio, che su una gamba può essere vecchio di ore): il segnale
"vede" rumore di prezzo che il giorno dopo sparisce. **Non tradabile** finché non
è rifatto sui settlement ufficiali (schema `statistics`): stima Databento
GC 0,37 $ + HG 0,52 $ (≈0,9 $, 2010-2026). Lo stesso artefatto non può creare il
vantaggio dei calendari stagionali (tenute 20-90 gg, scelta su anni precedenti),
ma ne aggiunge rumore: anche quelli vanno ricontrollati sui settlement.

### Settlement ufficiali (schema statistics) — scaricati GC e HG 2010-2026 (~0,9 $)

Estrazione: `fetch_databento --extract-settlements` (ultimo settlement ricevuto per
contratto e data `ts_ref`; esclusi i segnaposto a prezzo 0, 416 su 21.577 righe GC
2010-14, bloccati dal controllo unità). Settlement vs chiusura UTC: **mediana 2 $/oz
= 200 $ a contratto** sull'oro. Il server manda ~1 MB/min: oro ~35 min, rame ~35 min.

1. **Carry anomalo sull'oro sui settlement: MORTO.** 71 operazioni (contro 217 sulle
   chiusure: il rumore generava i segnali), 31-38% vinte, negativo in entrambi i
   periodi e con ogni ritardo. Artefatto confermato.
2. **Correzione importante — primo giorno di avviso (first notice).** Metalli, grani
   e LE si scambiano anche nel mese di consegna; lì la scadenza vicina converge allo
   spot e lo spread "rolla" quasi meccanicamente, ma un conto retail deve uscire
   prima (IBKR chiude, altrimenti consegna fisica). Lo scanner usciva 5 gg prima
   dell'ultimo scambio, cioè dentro il mese di consegna. Ora `last_retail_day()`:
   uscita entro la fine del mese precedente la consegna (`TRADES_IN_DELIVERY_MONTH`).
   Prima della correzione l'oro sui settlement faceva 87% (+433 $) — tutto roll del
   mese di consegna, non accessibile.
3. Rifatto per settore (chiusure, regola corretta): metalli calendario 54,4% (W 51,7),
   +86 $, 8/9 anni; ≤2022 52,6% / +51 $ (non significativo), >2022 56,7% / +133 $;
   per prodotto rame 59% / +131 $ (n=701), oro 52% / +137 $, argento 40% / −165 $.
   Grani 52,7% (≤2022 45,6%, >2022 61,3%), bestiame 55,4% (≤2022 62,9%, >2022 45,4%):
   ancora dipendenti dal periodo. Petrolio e gas invariati (non toccati dalla regola).
4. **Oro stagionale con regola corretta**: chiusure 51,8% (caso); settlement 72,1%,
   +562 $, ma n=122 e tutto 2023-2025 (2019-2022 in perdita); correlazione col prezzo
   dell'oro 0,53, 81% short spread → è una piccola scommessa al rialzo sull'oro
   (contango in $ che si allarga col prezzo). **Nessun vantaggio strutturale sull'oro.**
5. **Rame — il candidato migliore di tutto il progetto finora.**
   - Stagionale, settlement: 74,0% (W 71,2), +264 $, n=1.031; anni 64-93% tranne
     **2024 34% / −432 $**. Chiusure: 58,9%, +131 $. 93% delle scelte = vendi vicina /
     compra lontana. Correlazione col prezzo del rame −0,41: vince 87% se il rame
     scende, 65,5% se sale (il rame è salito nel periodo: non è beta).
   - **Regola fissa senza stagionalità** (sempre vendi vicina / compra lontana, mesi
     attivi HKNUZ distanti 1-6 mesi), settlement: 4 finestre provate, tutte positive:
     90/60 64,5% +159 $; **120/90 69,6% +282 $ (anni in perdita 2/16)**; 180/90 70,4%
     +244 $; 240/120 64,3% +218 $. Chiusure: 120/90 63,5% +234 $ (3/16).
   - **Versione accessibile, solo coppia attiva vicina→successiva (5 trade/anno),
     1 spread**: settlement 120/90 59,7%, +69 $/trade, totale 5.280 $ in 16 anni,
     DD max 1.382 $, peggiore −798 $; 180/90 66,2%, +86 $, DD 628 $, peggiore −410 $.
     Chiusure: 63,6-64,9%, +97-99 $, DD 1.390-1.658 $. 90/60 debole (+22 $).
   - Limiti: guadagno per trade piccolo rispetto ai costi stimati (~35 $ a giro:
     raddoppiarli dimezza il risultato); finestre scelte fra 3-4 provate (post-hoc,
     ma tutte nello stesso verso); meccanismo economico non ancora spiegato
     (ipotesi: premio per chi fornisce copertura sulla scadenza vicina + rientro
     delle tensioni di scorte). Prossimo: spread bid/ask e margini reali IBKR sullo
     spread HG (TWS paper, sola lettura), poi paper trading.

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
