# Arbitraggio sul funding rate — analisi e framework di backtest

Studio di fattibilità su dati reali: un bot delta-neutral che incassa il *funding
rate* dei perpetual, invece di scommettere sulla direzione del prezzo.

Nessun modulo qui dentro piazza ordini o usa API key. Legge dati di mercato
pubblici e produce numeri. La separazione è voluta: lo scopo di questa fase è
scoprire **se l'edge sopravvive ai costi** prima che del capitale — anche di
testnet — venga puntato addosso.

---

## Verdetto

1. Lo **spread di funding cross-venue** (Hyperliquid ↔ OKX su BTC/ETH) **non
   copre le commissioni retail**. Misurato su 96 giorni: ogni configurazione
   testata perde denaro, con win rate fra l'11% e il 50%.
2. Il **cash-and-carry** (long spot + short perp) **funziona**, ma rende
   **1.5-2% APR sul capitale** a leva 1×. È genuinamente delta-neutral — il
   96-99% del PnL viene dal funding, non dal prezzo — ma è un rendimento sotto
   il risk-free. Alzare la leva non aumenta l'edge, aumenta solo il rischio di
   liquidazione, e in questo campione la gamba short salta già sopra **2.0× su
   BTC e 1.2× su ETH**.
3. **Il segnale non serve a niente.** La versione passiva ("entra e non toccare")
   rende **nove volte** quella selettiva, perché ogni trade in più costa 38bp e
   l'EWMA non prevede il funding abbastanza bene da ripagarli. Quando l'edge è un
   premio strutturale e non una previsione, ottimizzare le entrate è controproducente.
4. **L'edge sta nella selezione trasversale, non nel timing.** La classifica del
   funding fra coin *persiste* (ρ Spearman ≈ 0.65, mai negativa in 99 periodi), e
   un book che tiene i 5 carry più ricchi rende **+2.84% APR** con il 96%+ del PnL
   attribuibile al funding. **141 configurazioni su 144** testate sono positive:
   è robustezza, non un picco di overfitting. Questa è anche l'unica
   configurazione dello studio in cui il limite inferiore di Wilson supera il 50%.
5. Il vincolo "win rate costante > 50%" **non è il vincolo giusto** e, con i
   volumi di trade che queste strategie generano, quasi mai è verificabile
   statisticamente. Sotto c'è la metrica da usare al suo posto.

**Il numero da portarsi via:** l'edge è reale, strutturale e non direzionale, ma
vale **1-3% APR sul capitale** — e due progetti indipendenti che hanno fatto lo
stesso esercizio con più dati arrivano allo stesso posto (§10): uno conclude che
il carry è l'unica di sette strategie a sopravvivere all'aritmetica dei costi, e
che perde comunque contro un titolo di stato.

Il campione qui (marzo 2025 – settembre 2026) cade interamente dentro il periodo
in cui il backtest di riferimento su sei anni **non apre nessuna posizione**,
perché il funding è sceso sotto la sua soglia. Non è una strategia diversa: è la
stessa strategia misurata nel regime magro.

---

## 1. Che cos'è il funding, e perché è un edge strutturale

Un perpetual future non ha scadenza, quindi niente lo obbliga a convergere verso
lo spot. Gli exchange lo ancorano con un pagamento periodico fra le due parti:

- **funding positivo** → i long pagano gli short;
- **funding negativo** → gli short pagano i long.

Chi incassa non sta indovinando il prezzo. Sta venendo pagato per fornire il lato
impopolare del book. Questo è il punto che rende la strategia diversa da tutto il
resto del repo: **l'edge non è una previsione, è un premio di rischio**.

E il premio ha un segno strutturale: nel cripto il retail a leva è quasi sempre
net long, quindi il funding è positivo la maggior parte del tempo. Misurato su
questo campione (96 giorni, media annualizzata):

| Venue | BTC | ETH |
|---|---|---|
| Hyperliquid perp | +5.96% APR (mediana +10.95%) | +6.19% APR (mediana +10.95%) |
| OKX perp | +5.45% APR (mediana +5.66%) | +3.93% APR (mediana +3.90%) |

Entrambi positivi su entrambi gli asset. Questo è il numero da cui parte tutto.

### Le due varianti, e perché non sono la stessa cosa

**A) Spread cross-venue** — long perp sul venue col funding basso, short perp sul
venue col funding alto. Delta ≈ 0. Ma la gamba long *paga* funding, quindi
incassi solo la **differenza**:

```
PnL_funding = Q × (funding_short_venue − funding_long_venue)
```

**B) Cash-and-carry** — long *spot* + short perp. Lo spot non paga funding,
quindi incassi il **livello pieno**:

```
PnL_funding = Q × funding_perp
```

Sui dati sopra: la differenza media |A| è ~5-6% APR, il livello B è ~5.5% APR su
OKX. Sembrano simili, ma A ha segno instabile (oscilla attorno allo zero e si
inverte di continuo) mentre B ha segno strutturalmente positivo. Per una
strategia che deve restare in posizione per settimane, questa è tutta la
differenza del mondo.

---

## 2. Perché "win rate > 50%" è il vincolo sbagliato

Un win rate del 95% si costruisce in dieci minuti: take-profit a 0.1%,
stop-loss a 5%. Vinci quasi sempre, e la coda ti porta via il conto. Il win rate
da solo non dice niente perché non vede la *dimensione* delle vincite.

Quello che decide se fai soldi è l'**expectancy netta**:

```
E = win_rate × avg_win − loss_rate × avg_loss − costi
```

C'è un secondo problema, più sottile e più serio per il tuo caso. Una strategia
di carry apre pochi trade — decine all'anno, non migliaia. Con 10 trade, un win
rate osservato del 70% ha un intervallo di confidenza al 95% che va da **~40% a
~89%**: è indistinguibile da una monetina. "Costante > 50%" è un'affermazione
sul **limite inferiore** dell'intervallo, non sulla stima puntuale.

Per questo `metrics.py` riporta, accanto al win rate, un **intervallo di Wilson**
e il flag `beats_coinflip`, vero solo se il limite inferiore al 95% sta sopra il
50%. È il campo da leggere prima di credere a tutti gli altri.

Servono circa **90 trade** perché un win rate vero del 60% produca un limite
inferiore sopra il 50%. Un carry con hold di 2 settimane ne fa ~25 all'anno. La
conseguenza è diretta: *il tuo criterio non è verificabile su un solo anno di
dati per questa famiglia di strategie*. Va sostituito con expectancy positiva +
attribuzione del PnL coerente (vedi §5).

---

## 3. L'aritmetica che decide tutto: il breakeven

Ogni ciclo apre e chiude **quattro ordini taker**: gamba long in, gamba short in,
gamba long out, gamba short out.

| | fee taker | 4 gambe + slippage 2bp/ordine |
|---|---|---|
| HL perp + OKX perp | 4.5 + 5.0 bp | **27 bp round trip** |
| OKX spot + OKX perp | 10.0 + 5.0 bp | **38 bp round trip** |

Lo spot costa il doppio del perp: è un'asimmetria che conta e che il modello
tiene per venue, non a forfait.

Quei basis point sono un **costo fisso**, mentre il funding si accumula nel
tempo. Quindi la domanda non è "quanto è largo lo spread" ma "**quanto a lungo
devo tenerlo perché lo spread paghi il round trip**":

```
breakeven_APR = (round_trip_bps / 10_000) × (8760 / ore_di_hold)
```

| hold | breakeven @27bp | breakeven @38bp |
|---|---|---|
| 8 ore | 295.7% APR | 416.1% APR |
| 1 giorno | 98.6% APR | 138.7% APR |
| 3 giorni | 32.9% APR | 46.2% APR |
| 7 giorni | 14.1% APR | 19.8% APR |
| 14 giorni | 7.0% APR | 9.9% APR |
| 30 giorni | 3.3% APR | 4.6% APR |
| 90 giorni | 1.1% APR | 1.5% APR |

**Questa tabella è l'intera strategia.** Lo spread medio misurato è ~5% APR:
significa che sotto le due settimane di hold non esiste nessuna configurazione
profittevole, qualunque sia il segnale. Il bot non deve essere bravo a prevedere,
deve essere bravo a **non fare churn**. È l'opposto esatto di come è impostato
`bot_engine.py` oggi, che ragiona su cooldown di 15-30 minuti.

---

## 4. Come funziona il backtest

Tre cose che questo motore si rifiuta di fingere.

**Il funding è riprodotto come eventi discreti**, ai timestamp in cui i venue
l'hanno effettivamente regolato, non come media oraria ricampionata. Una
posizione che attraversa tre settlement OKX viene pagata tre volte, non di più.
Il credito è `(entry, exit]`: entrare esattamente sull'orario di settlement non
incassa quel settlement.

**La copertura è imperfetta.** Le due gambe sono marcate sui prezzi dei
rispettivi venue, quindi il residuo (*basis PnL*) finisce nei risultati come
voce separata. Se il PnL di una strategia "market-neutral" è per lo più basis,
non è market-neutral: è una direzionale mascherata. Il report stampa la quota di
carry sul lordo e alza un warning sotto il 60%.

**Il margine è isolato per venue.** La gamba perdente può essere liquidata
mentre quella vincente è in profitto sull'altro exchange: i due conti non si
parlano. È così che saltano davvero i book di carry, quindi è modellato
esplicitamente invece che assunto via.

### Anti-lookahead

Al tempo di griglia `t` la strategia vede solo funding **già regolato**
(`event_time <= t`). Il forward-fill dell'ultimo valore noto è lecito;
l'interpolazione fra due settlement no, perché fa filtrare all'indietro la
stampa successiva. `test_annualized_at_never_reads_the_future` e
`test_forecast_only_uses_settled_prints` bloccano questa regressione.

### La trappola della normalizzazione

Hyperliquid regola il funding **ogni ora**, OKX **ogni 8 ore**. Un tasso dell'1bp
orario e uno dell'1bp ogni 8 ore sono lo stesso numero grezzo e **differiscono di
8 volte** annualizzati. Confrontarli senza normalizzare gonfia lo spread apparente
di un fattore 8 — è il modo più facile in assoluto di backtestare una strategia
che non esiste. Tutti i segnali passano da `annualize()`, e l'intervallo viene
**rilevato dai dati** (mediana dei gap) invece che dato per buono, perché alcuni
venue usano intervalli diversi per simbolo.

### Il segnale

Una sola convinzione: **gli spread di funding sono persistenti**. Un venue che
paga 30% annualizzato più di un altro da giorni probabilmente lo farà anche
domani, perché lo squilibrio è guidato da cose lente (chi tiene i long a leva,
quale venue usa il retail, dove stanno i desk di basis). Non è una previsione di
prezzo.

Quindi la previsione è la più stupida che rispetti quella convinzione: **media
esponenziale dello spread già regolato**. Niente regressioni, niente ML. Con
~2300 osservazioni orarie qualunque modello più ricco overfitta, e la versione
onesta rende interpretabile il verdetto del backtest.

---

## 5. Risultati misurati

Finestra: **96 giorni** (limite dello storico candle OKX, non una scelta).
Notional 10.000 $/gamba, leva 3×, slippage 2bp/ordine.

### A) Spread cross-venue — Hyperliquid ↔ OKX

Distribuzione dello spread annualizzato |OKX − HL|:

| | media | p90 | p99 |
|---|---|---|---|
| BTC | 4.85% | 10.16% | 15.43% |
| ETH | 6.19% | 12.10% | 16.74% |

Confrontala con la colonna @27bp della tabella di breakeven. Lo spread supera il
breakeven a 14 giorni (7.0%) solo nel ~10-20% delle ore — e deve **persistere**
per quelle due settimane, non solo toccare il livello un istante.

Sweep su soglia d'ingresso × hold minimo (96g, BTC ed ETH):

```
min_hold=48h (breakeven 49.3% APR)      min_hold=168h (breakeven 14.1% APR)
coin  entry  trades  win%   expect$      coin  entry  trades  win%   expect$
BTC      2%       9  11.1%   -19.91      BTC      2%       6  16.7%   -16.73
BTC      5%       3  33.3%   -13.27      BTC      5%       3  33.3%   -12.46
BTC      8%       1   0.0%   -26.88      BTC      8%       1   0.0%   -26.66
BTC     12%       0     —         —      BTC     12%       0     —         —
ETH      2%       5  20.0%    -5.35      ETH      2%       5  20.0%    -5.91
ETH      5%       6  16.7%   -14.06      ETH      5%       6  16.7%   -13.61
ETH      8%       2  50.0%    -1.80      ETH      8%       2  50.0%    -4.57
ETH     12%       0     —         —      ETH     12%       0     —         —
```

**Ogni singola configurazione con almeno un trade ha expectancy negativa.** Sopra
il 12% di soglia il bot non apre nulla, perché lo spread non arriva mai così in
alto in modo persistente — ed è il comportamento corretto: non fare nulla è
meglio che pagare 27bp per incassarne 10.

Conclusione: **lo spread di funding HL↔OKX su BTC/ETH non è un'opportunità a
livelli di fee retail.** Non è un problema di tuning dei parametri, è un problema
di aritmetica: l'edge lordo è più piccolo del costo di accesso.

### B) Cash-and-carry — long spot OKX / short perp OKX

Configurazione: long spot OKX (1×, notional pieno) / short perp. Costo 38bp
round trip (lo spot taker costa il doppio del perp).

| Config | leva short | campione | trade | funding | fee | netto | **APR su capitale** | liq. |
|---|---|---|---|---|---|---|---|---|
| OKX spot/perp, **selettivo** (entry 3%) | 3× | 96g | 2 | +$138.14 | −$126.00 | +$7.87 | **+0.22%** | 1 |
| OKX spot/perp, **passivo** | 3× | 96g | 1 | +$142.54 | −$88.00 | +$53.49 | **+1.53%** | 1 |
| OKX spot/perp, **passivo** | 1× | 96g | 1 | +$142.74 | −$38.00 | +$103.40 | **+1.97%** | 0 |
| spot OKX / perp **Hyperliquid**, passivo — BTC | 1× | 208g | 2 | +$277.80 | −$74.00 | +$194.56 | **+1.70%** | 0 |
| spot OKX / perp **Hyperliquid**, passivo — ETH | 1× | 208g | 2 | +$321.43 | −$74.00 | +$235.56 | **+2.06%** | 0 |

(Notional 10.000 $/gamba. La variante Hyperliquid ha un campione più lungo perché
lo storico candle *spot* di OKX arriva più indietro di quello perp.)

**Il carry è positivo, ed è davvero carry.** L'attribuzione lo conferma: il 96-99%
del PnL lordo viene dal funding, il residuo di basis è sotto i 12 $ su 10.000 $ di
notional. Non è una direzionale travestita. Ma il numero è piccolo: **1.5-2% APR
sul capitale**, sotto un titolo di stato.

### Tre cose che i numeri dicono e l'intuizione no

**1. La selettività distrugge il carry.** Il confronto più istruttivo della
tabella è la prima riga contro la terza. La versione "intelligente", che entra
solo quando il funding supera il 3% annualizzato ed esce quando decade, rende
**+0.22% APR**. La versione stupida, che entra e non si muove più, rende
**+1.97% APR** — nove volte tanto.

Il motivo è nella diagnostica del segnale:

```
entry selectivity     0.83x      (selettivo)
entry selectivity     0.10x      (passivo)
```

La *selettività* è lo spread medio al momento dell'ingresso diviso lo spread medio
complessivo. Un segnale che funziona dà un valore **sopra 1**: entra quando le
condizioni sono migliori della media. Il nostro dà **0.83** — entra quando sono
leggermente *peggiori*. L'EWMA non prevede niente: il funding di domani non è
predetto da quello di ieri abbastanza bene da battere i 38bp che costa agire sulla
previsione.

Questa è la lezione generalizzabile, e vale ben oltre il funding: **quando l'edge
è un premio strutturale e non una previsione, ogni trade in più è puro costo.**
Il bot ottimale non è quello che decide meglio, è quello che decide di meno.

**2. La leva non compra rendimento, compra liquidazioni.** Alzare la leva sulla
gamba short non aumenta il funding incassato di un centesimo: riduce solo il
capitale immobilizzato, quindi gonfia l'APR per divisione. Il prezzo è il rischio
di liquidazione, e in questo campione è enorme:

| | max calo da un massimo | max rialzo da un minimo |
|---|---|---|
| BTC | −11.8% → liquida la gamba **long** a ≥ 8.5× | **+49.5%** → liquida la gamba **short** a ≥ **2.0×** |
| ETH | −14.8% → liquida la gamba **long** a ≥ 6.8× | **+82.9%** → liquida la gamba **short** a ≥ **1.2×** |

In una finestra rialzista è la gamba *short* a saltare. Su ETH salta **sopra
1.2×**: in pratica qualunque leva. Il backtest lo cattura — la riga "passivo, 3×"
ha una liquidazione, che da sola costa 50 $ di penale e taglia l'APR da 1.97% a
1.53%. E il modello è ancora ottimista: al momento della liquidazione chiude
entrambe le gambe, mentre nella realtà l'exchange chiude solo quella perdente e ti
lascia **nudo e direzionale** sull'altro venue finché qualcosa non se ne accorge.

Questo è il vero vincolo della strategia, e non è nei numeri di rendimento: è il
motivo per cui i desk che fanno carry sul serio usano cross-margin, o tengono
margine di scorta pari a mesi di movimento avverso, o entrambi.

**3. Il vincolo del capitale è strutturale.** A 1× la posizione immobilizza
20.000 $ per 10.000 $ di notional esposto. Il funding sui major è ~5% APR *sul
notional*, che diventa ~2.5% *sul capitale*, meno i costi. Per alzarlo servono
solo due leve, entrambe con un prezzo: leva sulla gamba short (→ liquidazione,
punto 2) oppure asset con funding più ricco delle major (→ liquidità sottile e
slippage reale, non i 2bp assunti qui).

---

## 6. La selezione trasversale: dove l'edge c'è davvero

Il §5 ha stabilito che il funding di *una* coin non si può prevedere. La selezione
trasversale fa una scommessa diversa: non "il funding di questa coin è alto rispetto
alla sua storia", ma **"quali coin stanno pagando di più rispetto alle altre,
adesso"**. È una domanda meglio posta, perché la dispersione fra coin è enorme —
la leva retail si accalca su quello che si muove, non si distribuisce.

Snapshot dell'universo Hyperliquid (234 perp, 120 con OI > 1M $):

```
funding istantaneo annualizzato:  mediana +10.95%   p90 +100.1%   max +238.5%
```

La mediana sta sul *floor* del venue. Il novantesimo percentile è dieci volte tanto.

### Il test di falsificazione

La dispersione non basta: serve che la **classifica persista**. Se i leader di oggi
sono casuali domani, il book è solo un generatore di commissioni. Misurato su 180
giorni, 56 coin, finestre **non sovrapposte**:

| finestra | periodi | ρ Spearman | ρ min | top quintile | mediana | bottom quintile | Q1−Q5 |
|---|---|---|---|---|---|---|---|
| 3g | 59 | **0.669** | +0.39 | +13.6% | +8.6% | −8.6% | +22.2% |
| 7g | 24 | **0.649** | +0.48 | +12.1% | +8.0% | −7.1% | +19.2% |
| 14g | 11 | **0.672** | +0.55 | +12.1% | +7.9% | −7.1% | +19.2% |
| 30g | 5 | **0.639** | +0.48 | +12.3% | +8.2% | −2.8% | +15.1% |

**ρ ≈ 0.65 su ogni orizzonte, e mai negativo in nessuno dei 99 periodi.** Il
quintile più ricco va poi davvero a pagare il 12% annualizzato contro l'8% della
mediana. La premessa regge.

### Su tre anni, non su sei mesi

Il funding di Hyperliquid arriva indietro fino a **settembre 2023** — 26.279
settlement orari per coin, gratis e senza chiavi. E `persistence.py` usa *solo*
dati di funding, niente candele: il test si può quindi rifare sull'intero storico
senza altro. 39 coin, 1.095 giorni:

| finestra | periodi | ρ | ρ min | top quintile | mediana | bottom quintile | Q1−Q5 |
|---|---|---|---|---|---|---|---|
| 7g | **155** | 0.652 | +0.21 | **+26.7%** | +14.8% | −3.8% | +30.5% |
| 14g | 77 | 0.648 | +0.26 | +25.6% | +15.1% | −1.1% | +26.7% |
| 30g | 35 | 0.633 | +0.34 | +23.4% | +15.8% | +0.4% | +23.1% |
| 90g | 11 | 0.643 | +0.53 | +20.7% | +13.0% | +2.2% | +18.5% |

Due cose importanti. **ρ è identico** a quello dei 180 giorni (0.652 contro 0.649
a 7 giorni), su un campione otto volte più lungo e **mai negativo in 155 periodi**
indipendenti: la persistenza non era un artefatto della finestra recente.

E il funding storico è il **doppio** di quello recente: top quintile +26.7% contro
+12.1%, mediana +14.8% contro +8.0%. Il campione di 180 giorni su cui ho costruito
il book era il periodo *magro*, non quello favorevole. Il caveat che avevo scritto
al §6 ("questo campione vede il caso favorevole") era sbagliato nel verso opposto,
e i numeri sopra lo correggono.

> Un dettaglio che ha quasi falsato il risultato: la prima versione di `spearman()`
> non gestiva i **pareggi**. Il funding di Hyperliquid ha un floor dove decine di
> coin stanno a un valore *identico*; assegnare loro ranghi arbitrari per ordine di
> sort avrebbe prodotto lo stesso ordine in entrambe le finestre, fabbricando
> accordo dove i dati non ne hanno. Con i ranghi medi corretti il risultato non è
> cambiato (0.669 contro 0.670), quindi l'edge era reale — ma il test che lo ha
> scoperto vale più della conferma.

### Il book

`portfolio.py` classifica l'universo sul funding realizzato nelle ultime 168h e
tiene i primi N carry, ognuno individualmente delta-neutral (long spot / short
perp, stessa coin, stesso notional). Il portafoglio non ha direzione netta: varia
solo *quali* carry sono aperti.

Configurazione migliore, 180 giorni, 24 coin, 5 slot, leva 1×, slippage 5bp:

```
 BOOK
   rotations              14  (28/yr)
   utilisation            77.86% of slot-hours filled
   avg holding period     1201h (50.0 days)
   coins                  NEAR, LIT, ZEC, ENA, XPL, HYPE, TAO, PUMP, AAVE, ONDO

 P&L ATTRIBUTION
   funding collected      $2,220.33
   basis / hedge residual $13.76
   fees + slippage        $-836.00   (37.42% of gross carry)
   net                    $1,398.09

 RETURNS
   win rate               100.00%   [95% CI 78.47% - 100.00%]   SIGNIFICANT
   APR                    2.84%
   Sharpe (hourly, ann.)  1.92
   max drawdown           $204.06 (0.20%)
   liquidated legs        3
```

**È la prima configurazione di tutto lo studio in cui il limite inferiore
dell'intervallo di Wilson supera il 50%** — il criterio originale, finalmente
soddisfatto. Su un residuo di basis di 13.76 $ contro 2.220 $ di funding: non c'è
rischio direzionale dentro questo numero.

### Robustezza, non fortuna

Un solo risultato ottimizzato non significa niente. Sweep su **144
configurazioni** (slippage × slot × hold minimo × isteresi × lookback × frequenza):

| slippage/ordine | config positive | APR max | APR mediano | APR min |
|---|---|---|---|---|
| 2bp | **48/48** | +3.3% | +1.8% | +1.1% |
| 5bp | **48/48** | +2.9% | +1.4% | +0.7% |
| 10bp | **45/48** | +2.3% | +0.8% | −0.3% |

**141 configurazioni su 144 sono profittevoli**, con degrado monotono e ordinato
all'aumentare dei costi. È il profilo di un edge reale, non di un picco di
overfitting: se la profittabilità dipendesse da una combinazione fortunata di
parametri, la mediana sarebbe intorno a zero e solo la coda positiva.

E i driver sono tutti coerenti con la tesi "non fare churn":

| leva di parametro | rotazioni | utilizzo | fee/carry | APR medio |
|---|---|---|---|---|
| isteresi stretta (exit_rank 12) | 14.7 | 45% | 64% | +1.19% |
| isteresi larga (exit_rank 20) | **10.6** | **60%** | **52%** | **+1.79%** |
| lookback 336h | — | 38% | — | +1.10% |
| lookback 168h | — | **67%** | — | **+1.88%** |

Meno rotazioni, più capitale impiegato, meno commissioni. Ogni volta.

### Le tre cose che rovinano questo numero

**1. Il turnover, se lo lasci fare.** La mia prima configurazione — isteresi
stretta, uscita forzata a 21 giorni, 10bp di slippage — ha prodotto **fee pari al
131.9% del carry lordo** e un APR di −0.90%. Stesso segnale, stessi dati, segno
opposto. L'uscita forzata a scadenza era un default che avevo ereditato dalla
modalità single-pair: su un book che deve uscire per decadimento di rango, un
timer è costo puro. (Corretto: ora il default è 90 giorni in modalità portafoglio.)

**2. Lo slippage sulle alt, che non posso misurare.** Il 2bp è ragionevole su BTC,
non su un perp da 5M $ di open interest. La tabella sopra dice che a 10bp la
mediana scende a +0.8% e tre configurazioni vanno sotto zero. **Questa è la
variabile che decide se la strategia esiste, e i dati storici a candela non
possono darmela**: servono fill reali su testnet.

**3. Le liquidazioni restano, anche a leva 1×.** Tre gambe su quattordici sono
state liquidate — a leva 1× sulla gamba short, cioè servivano rialzi sopra il
+99%. Sulle alt succede. Non è un problema di leva mal scelta: **sull'universo alt
non esiste un'impostazione di leva che elimini il rischio di liquidazione**, perché
la coda dei rendimenti non è limitata. Si mitiga solo con cross-margin, margine di
scorta, o rinunciando alle coin più volatili — cioè proprio quelle che pagano.

### Bias dichiarati

- **Survivorship.** L'universo è selezionato sull'open interest di *oggi*. Le coin
  delistate o collassate nei 180 giorni non ci sono, e quelle diventate grandi sono
  incluse fin dall'inizio. Corregge verso l'alto, e non so di quanto.
- **Sweep in-sample.** Le 144 configurazioni sono valutate sullo stesso campione.
  Il numero da credere è la **mediana** (+1.4% a 5bp), non il massimo (+2.9%).
- **Campione corto.** 180 giorni, in una finestra che è stata rialzista (+32% BTC,
  +57% ETH). Il funding è strutturalmente più ricco nei mercati rialzisti: questo
  campione vede il caso favorevole.

---

## 7. Il costo di esecuzione, misurato invece che assunto

L'unico input non misurabile del backtest era lo slippage. Le candele storiche non
possono darlo: una barra oraria dice dove è andato il prezzo, non quanto sarebbe
costato spingere 10.000 $ attraverso il book in quel momento. Quindi lo portavo
come ipotesi — e l'intero verdetto ci oscillava sopra.

**Non serve la testnet per risolverlo, e la testnet non lo risolverebbe.** Un book
di testnet è una manciata di ordini sintetici: percorrerlo misura la testnet, non
il mercato. Il book *pubblico* di mainnet è la fonte onesta, è gratuito, non
richiede chiavi e leggerlo non rischia niente.

`liquidity.py` percorre i book live e calcola il costo effettivo delle quattro
attraversate di un carry (compra spot + vendi perp per aprire, vendi spot +
compra perp per chiudere). Snapshot su 10.000 $ per gamba:

| coin | round trip | spot buy | spot sell | perp sell | perp buy | profondità |
|---|---|---|---|---|---|---|
| BTC | **0.1bp** | 0.1 | 0.1 | 0.0 | 0.0 | 3.8M $ |
| ETH | **1.1bp** | 0.5 | 0.5 | 0.0 | 0.0 | 6.0M $ |
| HYPE | **1.9bp** | 0.1 | 1.3 | 0.5 | 0.1 | 85k $ |
| ZEC | **2.9bp** | 0.3 | 2.5 | 0.0 | 0.0 | 634k $ |
| TAO | 10.1bp | 4.8 | 1.4 | 1.6 | 2.3 | 54k $ |
| AAVE | 15.9bp | 8.7 | 5.4 | 0.9 | 0.9 | 74k $ |
| LIT | 22.9bp | 2.9 | 7.8 | 7.0 | 5.3 | 7k $ ⚠ |
| NEAR | 27.6bp | 2.7 | 2.7 | 9.1 | 13.1 | 34k $ |
| ONDO | 38.9bp | 4.8 | 2.9 | 13.0 | 18.2 | 26k $ |
| ENA | 40.1bp | 4.5 | 3.3 | 16.9 | 15.5 | 55k $ |
| XPL | 80.3bp | 11.6 | 8.3 | 31.6 | 28.8 | 32k $ |
| PUMP | **488.4bp** | 5.2 | 5.1 | 250.3 | 227.8 | 1k $ ⚠ |

⚠ = il book non assorbe la size; il numero è un limite inferiore, non una misura.

**Il rapporto fra il coin più economico e il più caro è di 4.000 volte.** Un'unica
cifra piatta di slippage è quindi l'input più fuorviante possibile in una strategia
trasversale — perché il ranking, lasciato a sé, seleziona proprio le coin sottili:
un tasso è alto in parte *perché* nessuno vuole stare dall'altra parte.

### Il risultato con i costi veri

`CostBook` porta il costo misurato per coin dentro il motore, e
`--max-slippage-bps` esclude quelle fuori budget:

```
escluse per liquidità: LIT (book too thin), PONS (book too thin),
                       VVV (book too thin), XPL (44bp round trip)
universo tradabile: 20 coin

   rotations              12  (24/yr)
   utilisation            67.14%
   funding collected      $1,724.75
   basis / hedge residual $71.24
   fees + slippage        $-721.71   (40.18% of gross carry)
   net                    $1,074.28

   win rate               83.33%   [95% CI 55.20% - 95.30%]   SIGNIFICANT
   APR                    2.18%
   Sharpe (hourly, ann.)  2.08
   liquidated legs        3
```

**+2.18% APR** con costi misurati, contro +2.84% con l'ipotesi piatta a 5bp. Il
limite inferiore di Wilson resta sopra il 50% (55.2%). La differenza fra i due
numeri — circa 65bp di APR — è il prezzo di aver smesso di indovinare.

> Un secondo bug di misura trovato qui: la prima versione troncava **entrambi** i
> book a 20 livelli. Hyperliquid ne serve 20 e basta (è un limite reale del venue),
> ma OKX ne serve 400, e i suoi book spot hanno prezzi molto granulari. Leggerne 20
> su PUMP significava vedere 0 $ di profondità invece di 412.000 $. Lo slippage
> delle alt risultava enormemente sovrastimato.

---

## 8. La distribuzione storica dello slippage

Il §7 misura il costo da un book *live*: risponde a "quanto costa adesso" e a
nient'altro. Il consiglio che ne era seguito — campionare con un cron per un mese
— era **sbagliato, o almeno inutile**. Binance pubblica archivi giornalieri
`bookDepth` dal 2023, gratuiti e senza autenticazione, che coprono ogni simbolo di
questo universo. La distribuzione non va aspettata: va scaricata.

Ogni file contiene, ogni 30 secondi, il **notional cumulato** entro 0.2%, 1%, 2%,
3%, 4% e 5% dal mid, su entrambi i lati. È esattamente l'input per prezzare un
ordine: si cammina verso l'esterno finché il cumulato copre la size, e la distanza
media percorsa è lo slippage.

Round trip su una gamba perp, ultimi 14 giorni, 40.320 snapshot per simbolo:

| simbolo | $10k mediana | p90 | p99 | peggio | p99/mediana |
|---|---|---|---|---|---|
| BTCUSDT | 0.0bp | 0.2 | 0.5 | 0.5 | *gratis* |
| ETHUSDT | 0.0bp | 0.4 | 0.6 | 0.6 | *gratis* |
| NEARUSDT | 0.9bp | 1.2 | 1.6 | 17.0 | 1.8× |
| TAOUSDT | 1.0bp | 1.6 | **295.1** | 335.8 | **296×** |
| ONDOUSDT | 1.1bp | 23.4 | 80.4 | 382.1 | 71× |
| WLDUSDT | 1.4bp | 2.6 | 2.6 | 14.2 | 1.9× |
| ENAUSDT | 1.4bp | 27.9 | **277.3** | 318.2 | **193×** |
| AAVEUSDT | 1.5bp | 3.1 | 9.8 | 15.6 | 6.6× |

**La colonna che conta è l'ultima.** Le mediane sono tutte sotto i 2bp e sembrano
rassicuranti; ma su TAO ed ENA il 99° percentile è **200-300 volte** la mediana. E
un carry non si chiude in un momento mediano: si chiude quando il funding si
inverte, cioè esattamente quando la profondità è sparita. Dimensionare sulla
mediana significa dimensionare sul caso che non ti capiterà mai quando serve.

Nota che NEAR e WLD hanno stress multiple di 1.8-1.9× mentre TAO ed ENA stanno a
200-300×. **Non è una proprietà della liquidità mediana** — hanno mediane quasi
identiche — ma della fragilità del book. È una dimensione di rischio che nessuna
delle metriche precedenti vedeva.

### Due bug trovati facendo questo

**Il parser scartava tutto.** Il campo `percentage` è scritto `"-5.00"`, non
`"-5"`, quindi `int()` sollevava e ogni riga veniva saltata in silenzio: zero
snapshot su 2.880. E c'è una banda a **0.2%** oltre a quelle intere — trattare la
prima banda come larga 1% sovrastima di cinque volte il costo di un ordine piccolo.

**L'archivio contiene giornate corrotte.** Dal 7 all'11 settembre 2026 NEARUSDT
riporta **13 $ di notional identici su tutte e sei le bande**. Un book reale si
approfondisce allontanandosi dal mid: il cumulato a 5% non può uguagliare quello a
0.2%. Lasciate dentro, quelle righe si travestono da crisi di liquidità — da sole
producevano un "30% degli snapshot non assorbe 10.000 $" su un perp che scambia
centinaia di milioni al giorno. `is_plausible()` le rifiuta, e dopo il filtro
NEAR passa da 30.3% a **0.0%** di snapshot non coperti.

### Limiti onesti

- **È Binance, non Hyperliquid/OKX.** Profondità e spread correlano fra venue ma
  non coincidono. L'uso corretto è prendere da qui la **forma** della distribuzione
  (il rapporto p99/mediana) e il **livello** da `--liquidity` sui venue davvero
  scambiati.
- **Non vede lo spread dentro la prima banda.** Per ordini che non escono dallo
  0.2% il costo risulta quasi nullo, mentre il mezzo spread si paga comunque. Il
  modulo serve alle size che vanno oltre il touch, non a quelle minuscole.

---

## 9. Il backtest lungo, e la cosa che non avevo modellato

Hyperliquid dà 3 anni di funding ma solo 208 giorni di candele, quindi un carry
HL non si può backtestare a lungo con il basis misurato. MEXC sì: **funding da
aprile 2025, candele perp dal 2023, spot dal 2023**, tutto raggiungibile senza
restrizioni geografiche. Spot e perp sullo stesso venue significa che il basis è
quello vero, non un proxy.

540 giorni, stesso motore, stessi parametri. Cambia solo l'universo:

| universo | funding | basis | fee | netto | APR | liquidazioni | win rate |
|---|---|---|---|---|---|---|---|
| 24 coin, **incluse le alt sottili** | +$1.933 | **−$1.322** | −$894 | −$283 | **−0.19%** | 7 | 68.8% (non significativo) |
| 21 coin, **solo liquide** | +$2.460 | **−$70** | −$966 | **+$1.424** | **+0.96%** | 3 | **75% [CI 55.1–88.0] SIGNIFICATIVO** |

**Il residuo di copertura migliora di 19 volte.** Ed è tutto lì: il funding
incassato è simile, le commissioni sono simili, quello che ribalta il segno è il
basis.

### Perché: la copertura statica si sfalda quando il prezzo corre

Il trade peggiore è PONS, −$1.259 di basis su una gamba da 10.000 $ in 81 ore.
Ho controllato i marks a mano, perché un numero così su spot e perp *dello stesso
venue* sembrava un bug:

```
entrata:  spot 0.00973   perp 0.00940    perp 3.4% SOTTO lo spot
uscita:   spot 0.01867   perp 0.01922    perp 2.9% SOPRA lo spot
gamba long  +91.9%   gamba short  −104.5%   somma  −12.6%
```

Il rapporto perp/spot si è mosso del **6.5%**. La perdita è del **12.6%**. Il
fattore due non è un errore: **il prezzo è quasi raddoppiato durante l'hold**, e
in una coppia aperta a notional uguale le due gambe si sbilanciano man mano che
il prezzo corre. Un movimento del basis si applica a una posizione che nel
frattempo è cresciuta, quindi il costo in dollari si amplifica.

Non è un difetto del backtest — un controllo di invarianza su ogni trade chiuso
conferma che il basis registrato coincide al centesimo con quello implicito nei
marks. È **l'assenza di ribilanciamento della copertura**, che questo motore non
fa e che i desk veri fanno. Le alternative sono due, entrambe con un prezzo:
ribilanciare (paghi commissioni a ogni aggiustamento) oppure escludere le coin
che possono raddoppiare durante un hold — cioè, di nuovo, proprio quelle che
pagano il funding più alto.

E le liquidazioni di PONS sono reali, non artefatti: il token è più che
raddoppiato **tre volte** in 540 giorni, e a leva 1× la gamba short salta sopra
il +99.5%.

### La qualità della copertura è una dimensione a sé

`liquidity.py` e `depth_history.py` chiedono entrambi *quanto costa attraversare
il book*. Nessuno dei due vede se la copertura tiene. Una coin può avere un book
perfettamente servibile e un perp che deriva dallo spot. `basis.py` misura la
cosa che finisce davvero nel PnL: non il livello del basis ma il suo **cambiamento
durante l'holding period** — entri a un basis, esci a un altro, la differenza è
tua che tu la voglia o no.

Movimento del basis su un hold di 20 giorni, MEXC perp vs spot, 540 giorni:

| coin | p99 | mesi di carry persi |
|---|---|---|
| ETH | 0.07% | 0.0 |
| SOL, XRP, DOGE, LTC, SUI | 0.10–0.16% | 0.1 |
| NEAR, ENA, TAO, AAVE, ONDO | 0.23–0.27% | 0.1–0.2 |
| XPL | 0.43% | 0.3 |
| VVV, LIT | 0.83–0.95% | 0.5–0.6 |
| **HYPE** | **1.66%** | **1.0** |

22 coin su 23 tengono una copertura degna del nome (p99 sotto l'1%). Un livello
di basis costante non costa niente — entri ed esci allo stesso; è il movimento
che paghi.

### Nota metodologica: il throttling stava falsando l'universo

Durante questi run MEXC ha iniziato a rispondere 403 a metà caricamento, e
`load_universe` scartava in silenzio le coin in volo — BTC, AAVE, BCH e DOT sono
sparite da un run. Un universo che dipende da quali richieste sono passate non è
riproducibile, e la selezione ne risulta distorta. Ora c'è un rate limiter per
host (`_HOST_MIN_INTERVAL`) e 7 tentativi con backoff limitato: pacing non per
cortesia, ma per correttezza.

---

## 10. Lavoro correlato, e cosa dice dei numeri qui sopra

Questa famiglia di strategie è battuta. Su GitHub ci sono decine di progetti di
funding arbitrage, ma quasi tutti sono **scanner o esecutori live** — trovano lo
spread e piazzano gli ordini. I framework di ricerca con un modello di costo
onesto sono pochissimi, ed è lì che il confronto diventa utile.

### Due progetti indipendenti arrivano alla stessa conclusione

**[aaronpascalkujur/trading-strategy-research]** ha testato sette strategie
contro l'aritmetica dei costi: MA crossover, mean reversion su tre orizzonti,
pair trade ETH/BTC, e il carry sul funding. **Il carry è l'unico sopravvissuto** —
esattamente lo stesso esito del §5 qui. Ma il suo verdetto finale è più duro del
mio: 11.57%/anno sul notional, che diventa **4.39% dopo tasse e vincoli di
capitale**, contro un titolo di stato indiano al **5.63% senza rischio di
liquidazione**. Il titolo del repo lo dice meglio di qualsiasi riassunto: *"five
disproved, one that works and still lost to a government bond"*.

**[zwmjj/funding-rate-arb]** ha backtestato il carry long spot / short perp su
BTC ed ETH su Binance, da gennaio 2020 ad aprile 2026 — sei anni:

| | BTC | ETH |
|---|---|---|
| rendimento annuo lordo | 9.0% | 11.4% |
| trade | 18 | 14 |
| win rate | **100%** | **100%** |
| hold medio | 38g | 58g |
| max drawdown | −0.28% | −1.80% |

E, come al §2 qui, **trattano il win rate del 100% come un avvertimento, non come
una validazione**: *"la strategia non è stata testata — sta descrivendo un periodo
in cui il funding perpetuo è stato persistentemente positivo"*.

### Perché io misuro l'1% e loro il 9-12%

Non è una discrepanza: sono tre differenze tutte spiegabili, e la terza è la più
importante.

1. **Notional contro capitale.** Loro quotano sul notional. A leva 1× il carry
   immobilizza 2× notional (spot intero + margine perp), quindi 11.57% sul
   notional è ~5.8% sul capitale prima dei costi. Circa metà del divario sparisce
   qui.

2. **Basis e liquidazioni.** Nessuno dei due li modella. zwmjj lo dichiara:
   *"spot e perp sono trattati come coperture 1:1 all'ingresso; lo slippage
   effettivo è ignorato"*. Nel mio run a 540 giorni il basis costa −$70 e ci sono
   3 liquidazioni.

3. **Il regime.** Questa è la parte che conta. zwmjj riporta che **il 79% del PnL
   su BTC e l'82% su ETH viene da trade aperti prima del 2022**, e che nel 2025 il
   funding è sceso a 4.3%/anno — sotto la loro soglia d'ingresso — quindi hanno
   aperto **zero trade nel 2025 e nel 2026**.

   Il mio campione MEXC va da marzo 2025 a settembre 2026. **È interamente dentro
   il periodo in cui il loro backtest non apre posizioni.** Non sto misurando una
   strategia diversa: sto misurando la stessa strategia nel regime che gli altri
   backtest evitano semplicemente restando fuori.

   Il che rende coerente anche il §6: su 3 anni il top quintile ha reso +26.7%
   APR contro +12.1% negli ultimi 180 giorni. Il funding *era* più ricco prima.

### La versione trasversale esiste — nella forma che avevo scartato

**[nsheng1568/funding-dispersion-trade]** fa la selezione trasversale su ~30 coin
di Hyperliquid, ma nella variante **long-short direzionale**: lungo la coin col
funding più basso, corto quella col più alto, neutralizzando il beta via PCA su
BTC/ETH/SOL. È precisamente il trade che avevo descritto e scartato — cattura
l'intero spread Q1−Q5 (~19-30% APR nei miei dati) invece del solo livello, ma
introduce rischio di prezzo relativo fra panieri.

Il suo risultato: **~7% APR netto costi a ~25% di volatilità**, che l'autore stesso
definisce *"un rendimento corretto per il rischio altamente inappetibile"* —
Sharpe ~0.28. Il rischio di prezzo relativo si è mangiato il vantaggio, come
temevo.

Nota di convergenza indipendente: usa EWMA con half-life di **168h e 72h**. Io ero
arrivato a un lookback di 168h per il ranking. Stessa scala temporale, trovata
separatamente.

### Cosa nessuno di questi modella

| | scanner live | zwmjj | aaronpascal | nsheng | questo repo |
|---|---|---|---|---|---|
| costo di transazione quantificato | ~ | ✅ 16bp | ✅ | ✅ | ✅ misurato dai book |
| coda dello slippage (p99) | ✗ | ✗ | ✗ | ✗ | ✅ archivi dal 2023 |
| deriva del basis | ✗ | ✗ dichiarato | ✗ | ✗ | ✅ misurata |
| liquidazione a margine isolato | ~ | ✗ | ✗ | ✗ | ✅ per gamba |
| persistenza del rango di funding | ✗ | ✗ | ✗ | premessa non testata | ✅ ρ su 3 anni |
| walk-forward | ✗ | ✗ (sensitivity) | ✅ | ✗ | ✗ **manca** |
| ribilanciamento della copertura | alcuni | ✗ | ✗ | ✗ | ✗ **manca** |

Le due caselle vuote nell'ultima colonna sono esattamente i due punti aperti del
§12. E su una delle due la letteratura ha già una risposta utile.

### Sul ribilanciamento, la letteratura è scoraggiante

La ricerca sull'hedging dinamico dice che il **ribilanciamento a soglia batte
quello periodico** — si ottiene quasi tutto il beneficio con molti meno trade — e
che una soglia di deriva del ~15% fa scattare un aggiustamento circa una volta a
trimestre.

Ma l'aritmetica è brutale, e vale la pena riportarla per intero: *una posizione da
100.000 $ ribilanciata tre volte a settimana a 4bp di round trip costa ~600 $ al
mese; se il funding lordo vale 900 $ al mese, ne restano 300 prima di qualsiasi
movimento di basis o inversione del funding.*

Quindi il ribilanciamento non è un miglioramento gratuito: è uno scambio fra
deriva della copertura e commissioni, e a soglie strette il rimedio costa più
della malattia. Quando lo implementerò, il parametro da ottimizzare non è la
frequenza ma **la soglia di deriva**.

### Letteratura accademica

- La persistenza che misuro al §6 è corroborata: un'analisi ad alta frequenza su
  26 exchange (11 CEX, 15 DEX) trova che gli spread di funding mostrano
  *"persistenza estremamente elevata"*.
- Uno studio SSRN, *["Failure of Cross-Sectional Alpha Screening on Cryptocurrency
  Perpetual Futures"]*, trova che i segnali di funding **non** contengono alpha
  trasversale sfruttabile su 10 perp Binance a orizzonte 8h: Rank IC fra −0.0097 e
  +0.0243, Sharpe netto −2.9/−3.2, drawdown −95.6%.

  **Attenzione a non leggerlo come una smentita di questo lavoro.** Loro misurano
  l'IC di *funding → rendimento di prezzo* su portafogli long-short direzionali.
  Io misuro *funding → funding* su posizioni delta-neutral per coin. Sono due
  grandezze diverse, e il loro fallimento riguarda proprio la variante
  direzionale che ho scartato — la stessa che nsheng1568 porta a Sharpe 0.28.

[aaronpascalkujur/trading-strategy-research]: https://github.com/aaronpascalkujur/trading-strategy-research
[zwmjj/funding-rate-arb]: https://github.com/zwmjj/funding-rate-arb
[nsheng1568/funding-dispersion-trade]: https://github.com/nsheng1568/funding-dispersion-trade
["Failure of Cross-Sectional Alpha Screening on Cryptocurrency Perpetual Futures"]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6701738

---

## 11. Indicatori tecnici: no come segnale, sì come cancello di rischio

Domanda naturale: perché non combinare il carry con MACD, SuperTrend o simili, per
tenere la posizione solo nei periodi favorevoli? C'è un meccanismo plausibile
dietro — il funding è positivo quando i long a leva si accalcano, e questo
correla col trend di prezzo, quindi un indicatore *potrebbe* anticipare il
regime.

Non è un'opinione da dare: è misurabile. 21 coin, 540 giorni, 75 periodi
settimanali non sovrapposti, IC di rango trasversale contro il funding
effettivamente realizzato nei 7 giorni successivi.

| predittore del funding futuro | IC medio | IC min | IC max | periodi con IC > 0 |
|---|---|---|---|---|
| **il funding stesso (trailing 168h)** | **0.498** | −0.12 | 0.85 | **99%** |
| MACD histogram | 0.025 | −0.45 | 0.59 | 55% |
| volatilità realizzata 168h | 0.046 | −0.37 | 0.40 | 57% |
| momentum 168h | **−0.051** | −0.45 | 0.48 | 40% |
| prezzo vs SMA200 | **−0.067** | −0.62 | 0.43 | 31% |

**Nessun indicatore di prezzo prevede il funding.** MACD è una monetina;
momentum e SMA200 sono leggermente *negativi*, cioè girati nel verso sbagliato.
Il funding prevede sé stesso venti volte meglio di qualsiasi proxy.

Il motivo è quasi tautologico una volta detto: un indicatore di prezzo sarebbe un
*sostituto rumoroso* di una cosa che qui si osserva direttamente. Il funding non
va stimato — l'exchange lo pubblica. Aggiungere un proxy a un osservabile può solo
peggiorare.

E il §5 aveva già dato l'avvertimento: la versione "intelligente" con timing EWMA
aveva selettività 0.83 e rendeva un nono di quella passiva. Su questa strategia
qualunque strato di timing ha finora peggiorato le cose.

### Ma la domanda giusta non era quella

Gli indicatori non prevedono il *rendimento*. Prevedono però molto bene i due
**modi di fallire** da cui sono venute tutte le perdite del §9:

| predittore | IC vs rialzo avverso (liquida la gamba short) | IC vs movimento del basis |
|---|---|---|
| **volatilità realizzata 168h** | **0.356** | **0.363** |
| range 168h (stile SuperTrend) | 0.305 | 0.293 |
| momentum 168h | 0.018 | −0.049 |

Un ordine di grandezza più della loro capacità di prevedere il funding. E i
quintili di volatilità, su 1.514 osservazioni, dicono la cosa in modo brutale —
rialzo massimo su un hold di 20 giorni:

| quintile di volatilità | mediana | p90 | p99 | **oltre +99.5% = liquidazione a 1×** |
|---|---|---|---|---|
| Q1 (più calmo) | 7.5% | 24.5% | 63.4% | **0.00%** |
| Q2 | 10.0% | 41.9% | 84.3% | 0.34% |
| Q3 | 11.6% | 37.6% | 94.5% | 0.68% |
| Q4 | 15.1% | 58.0% | 166.0% | 3.04% |
| Q5 (più agitato) | 18.5% | 71.8% | 157.2% | **5.76%** |

Da **zero a 5.76%** di probabilità di liquidazione. Questa non è una relazione
marginale da sfruttare con cura: è una separazione netta, su un campione grande.

### Il filtro implementato, e i suoi limiti

`PortfolioParams.exclude_vol_quantile` scarta all'ingresso una frazione
dell'universo per volatilità trailing. Il gate è **trasversale** e non a soglia
assoluta, così mantiene senso quando la volatilità di tutto il mercato cambia. Si
applica solo all'ingresso: una posizione già aperta si lascia stare, perché
chiuderla costa un round trip intero e il movimento temuto è probabilmente già
avvenuto.

540 giorni, 21 coin, stesso motore:

| gate | rotazioni | utilizzo | basis | netto | APR | **liquidazioni** | win rate [CI low] |
|---|---|---|---|---|---|---|---|
| 0% | 10 | 22% | +$388 | +$327 | 0.22% | **4** | 60% [31%] |
| 10% | 10 | 12% | +$402 | +$777 | **0.53%** | 1 | 80% [49%] |
| 20% | 8 | 7% | +$287 | +$685 | 0.46% | 1 | 88% [53%] |
| 30% | 6 | 5% | +$248 | +$612 | 0.41% | **0** | 100% [61%] |
| 40% | 5 | 4% | −$72 | +$138 | 0.09% | 1 | 80% [38%] |

**Cosa credere e cosa no.** Le liquidazioni che scendono da 4 a 0 sono coerenti
con la tabella dei quintili, che poggia su 1.514 osservazioni: quello è il
risultato solido, ed è il motivo per cui il gate resta nel codice.

Il miglioramento di APR **non** va creduto come stima di rendimento. Sono 5-10
rotazioni, gli intervalli di Wilson si sovrappongono tutti, e la soglia è scelta
sullo stesso campione su cui è misurata — esattamente l'overfitting contro cui
mette in guardia il §2. Che il massimo cada al 10% e non al 30% è rumore.

E il prezzo del gate è visibile: **l'utilizzo crolla dal 22% al 5%**. Su un book
il cui problema principale era già il capitale inattivo, un filtro che dimezza le
posizioni aperte toglie da una parte quello che dà dall'altra. Il gate riduce il
rischio di coda; non fa guadagnare di più.

### In una riga

Gli indicatori tecnici non hanno niente da dire su *quanto* incasserai — quello
te lo dice il funding. Hanno molto da dire su *quando la copertura si romperà*, ed
è lì che vanno messi.

---

## 12. Rischi

### Modellati dal backtest

| Rischio | Come |
|---|---|
| **Liquidazione della gamba isolata** | Margine siloed per venue; force-close quando la perdita di una gamba supera `margine − maintenance`, più penale di 50bp |
| **Basis / hedge imperfetto** | Le gambe sono marcate sui prezzi dei rispettivi venue; il residuo è una voce separata del PnL |
| **Costi di transazione** | 4 gambe taker + slippage, per venue |
| **Inversione del funding** | Uscita su `spread_flip`, con hold minimo che impedisce di scappare troppo presto |

### NON modellati — e sono quelli che fanno male

| Rischio | Perché conta |
|---|---|
| **Rischio exchange** | Il capitale sta su due venue. Se uno congela i prelievi, la copertura non esiste più ma l'esposizione sì. Storicamente è la causa di perdita n°1 in questa strategia, e nessun backtest la vede |
| **Gamba nuda dopo la liquidazione** | Il modello chiude entrambe le gambe insieme. L'exchange chiude solo quella perdente: resti direzionale sull'altro venue finché non intervieni. È la lettura benevola di una liquidazione, quindi conta il *numero* di liquidazioni, non il PnL del trade liquidato |
| **Esecuzione non simultanea** | Fra il fill di una gamba e l'altra sei direzionale. Su un movimento veloce quel gap costa più di settimane di carry. `CostModel.execution_slip_bps` esiste per stimarlo ma il default è 0 |
| **Profondità del book** | Il backtest assume che 10k $ si eseguano al mid. Vero per BTC/ETH, falso per le alt dove il funding è più ricco |
| **Cambi di regime nelle fee** | I tier cambiano, i venue lanciano promo. A 27bp di margine, 1bp conta |
| **Auto-deleveraging / socialized loss** | La gamba short può esserti chiusa d'ufficio da eventi dell'exchange |
| **Rischio stablecoin** | Collaterale USDT/USDC su entrambe le gambe |

### Il punto sulla leva

Con leva `L` la gamba perdente salta su un movimento avverso di circa `1/L`. La
tentazione è dire "3× è prudente". I dati di questo campione dicono di no: il
rialzo massimo da un minimo è stato **+49.5% su BTC** e **+82.9% su ETH**, che
liquidano la gamba short rispettivamente sopra **2.0×** e **1.2×**.

Quindi la regola prudenziale non è un numero fisso, è: **la leva sulla gamba short
deve sopravvivere al più grande rialzo che l'asset può fare durante l'hold.** Su un
carry tenuto per mesi in cripto, quel numero è vicino a 1×. In una strategia il cui
rendimento lordo è il 5% annuo, una liquidazione cancella anni di carry.

Il margine va monitorato **in continuo**, non a ogni ciclo dell'LLM: una posizione
di carry può passare da sana a liquidata in minuti, e un loop che gira ogni pochi
minuti chiamando un modello non è un risk manager.

---

## 13. Uso

```bash
# 1. Il test di falsificazione: la classifica del funding persiste?
python -m research.funding_arb.run --persistence --days 180 --universe-size 58

# 2. Il book trasversale (la configurazione che funziona)
python -m research.funding_arb.run --portfolio --days 180 \
    --venue-a okx_spot --venue-b hyperliquid --universe-size 24 \
    --max-positions 5 --entry-rank 5 --exit-rank 20 --rank-lookback 168 \
    --min-hold 480 --leverage 1 --slippage-bps 5

# 3. Report completo su una singola coppia
python -m research.funding_arb.run --coins BTC ETH --days 365

# Cash-and-carry (la gamba a è spot: vietato lo short)
python -m research.funding_arb.run --coins BTC --days 365 \
    --venue-a okx_spot --venue-b okx --no-reverse \
    --min-hold 336 --max-hold 2160

# Sweep sulla soglia d'ingresso
python -m research.funding_arb.run --coins BTC --sweep-entry 0.03,0.05,0.08,0.12

# Export completo dei trade
python -m research.funding_arb.run --coins BTC --json /tmp/out.json

pytest research/funding_arb/tests/ -v
```

Venue disponibili: `hyperliquid`, `okx`, `bybit`, `binance`, `okx_spot`,
`binance_spot`. Una gamba spot viene automaticamente forzata a leva 1× (notional
pieno, nessuna liquidazione possibile): `--leverage` si applica solo alle gambe
perp.

> **Nota rete:** `api.bybit.com` e `fapi.binance.com` sono dietro CloudFront e
> bloccano diverse regioni. Gli adapter sono implementati e corretti, ma il
> fetch va lanciato da un host in una regione supportata (es. il tuo deploy
> Railway). Da questo container rispondono solo Hyperliquid e OKX — per questo
> il backtest qui sopra usa quella coppia.

Le risposte HTTP sono cachate in `.cache/` (gitignorata). `--no-cache` la svuota.

---

## 14. Runbook operativo

### Quello che NON serve

**TAAPI.** Verificato contro la loro API: `/candles` esiste (risponde 401 senza
chiave), ma un endpoint per il funding **non esiste** — `api.taapi.io/fundingrate`
risponde *"The endpoint (or indicator) you are calling does not exist"*. TAAPI è
un'API di indicatori tecnici sopra le candele: dà EMA, MACD, RSI e OHLCV, non
tassi di funding. Siccome il funding **è** l'edge, TAAPI non può fornire il dato
che conta. Gli indicatori sono comunque irrilevanti qui (§5: l'edge è un premio
strutturale, non un pattern prevedibile), e il piano free è limitato a 1 richiesta
ogni 15 secondi.

**Un abbonamento per lo storico.** Non c'era un problema di dati da risolvere:

| dato | fonte | profondità | costo |
|---|---|---|---|
| funding rate | Hyperliquid `/info` | **3 anni** (26.279 punti orari) | gratis |
| profondità del book | Binance `data.binance.vision` | **dal 2023**, ogni 30s | gratis |
| candele spot | KuCoin, Coinbase, Kraken, Gate.io, MEXC, Bitget | anni | gratis |
| candele perp HL | Hyperliquid `candleSnapshot` | 208 giorni (cap del venue) | gratis |

Il solo vincolo reale è l'ultima riga: le candele perp di Hyperliquid si fermano a
5.000 barre. Il funding — il dato che porta l'edge — non è mai stato il problema.

**Il cron di campionamento dello slippage.** Era il mio consiglio precedente ed
era inutile: gli archivi `bookDepth` danno già la distribuzione storica (§8).
`--sample-to` resta utile solo per HL/OKX specificamente, che non pubblicano
archivi, ma non è più sul percorso critico.

**La testnet, per validare l'edge.** Un book di testnet è una manciata di ordini
sintetici. Serve a una cosa sola, il punto 4.

### Quello che serve, in ordine

**1. Ridimensionare il book sul p99, non sulla mediana.**

```bash
python -m research.funding_arb.run --depth-history --depth-days 30 --notional 10000
```

Poi rifai il §6 escludendo le coin il cui p99 è insostenibile, non quelle la cui
mediana lo è. Su TAO ed ENA il p99 è 200-300× la mediana: sono le prime da tagliare
anche se oggi sembrano economiche.

**2. Estendere il backtest del book a 3 anni.** Il funding c'è; mancano le candele
perp oltre i 208 giorni. Due strade, in ordine di onestà:
   - portare la gamba perp su un venue con storico profondo e raggiungibile
     (MEXC ha 1.5 anni di funding e candele lunghe: `contract.mexc.com`);
   - oppure girare il backtest a 3 anni **solo su funding e commissioni**, con il
     basis posto a zero, e validarlo contro la finestra di 208 giorni dove il
     basis è misurabile. Il residuo misurato è stato +13.76 $ e +71 $ su ~2.000 $
     di funding, cioè 1-4% e **di segno positivo**: azzerarlo è un'approssimazione
     leggermente conservativa, non un trucco. Va comunque dichiarata.

**3. Walk-forward.** Le 144 configurazioni del §6 sono in-sample. Con 3 anni di
funding c'è finalmente abbastanza campione per tarare su una finestra e misurare
sulla successiva.

**4. Solo adesso la testnet, e solo per l'impianto.** Non per validare l'edge, ma
per verificare la meccanica: che le due gambe partano insieme, che gli
arrotondamenti di size passino, che il codice si accorga di un fill parziale.
`HYPERLIQUID_NETWORK=testnet` è già supportato in `config_loader.py`. Una sola
domanda: **quanti secondi passano fra il fill della prima gamba e quello della
seconda?** In quel buco sei direzionale, ed è l'unica variabile che né il book
pubblico né gli archivi possono darti. Misurala e mettila in
`CostModel.execution_slip_bps`.

**5. Poi, e solo poi**, capitale reale alla size minima.

### Un avvertimento sulla scala

A 500-5.000 $ questa strategia **non ha senso economico**. Serve margine su due
venue contemporaneamente, i size minimi sono ~10-20 $ per ordine, e il 2% annuo su
2.000 $ fa 40 $. La soglia in cui il carry ripaga l'infrastruttura è nell'ordine
delle decine di migliaia di dollari.

---

## Struttura

```
research/funding_arb/
├── venues.py     # adapter API pubbliche + cache su disco (HL, OKX, Bybit, Binance, spot)
├── costs.py      # modello di costo e curva di breakeven
├── dataset.py    # allineamento su griglia oraria causale, normalizzazione intervalli
├── strategy.py    # EWMA causale dello spread, regole entry/exit (single-pair)
├── backtest.py    # motore a eventi: funding discreto, basis, liquidazione isolata
├── persistence.py # il test di falsificazione: rho di Spearman e quintili
├── universe.py    # scoperta delle coin copribili (OI + esistenza dello spot)
├── portfolio.py   # book trasversale: classifica l'universo, tiene i primi N
├── liquidity.py   # costo di esecuzione reale, percorrendo i book live
├── depth_history.py # distribuzione storica del costo, dagli archivi Binance
├── basis.py       # qualita' della copertura: deriva perp-spot sull'holding
├── metrics.py     # expectancy, Wilson CI, Sharpe, drawdown, attribuzione PnL
├── run.py         # CLI (6 modalità: single-pair, --persistence, --portfolio,
│                  #      --liquidity, --depth-history, --basis)
└── tests/         # 68 test: lookahead, normalizzazione, contabilità del funding,
                   # pareggi nel ranking, aritmetica del book, righe corrotte,
                   # qualità della copertura
```

## Ordine di lettura consigliato

§3 (l'aritmetica del breakeven), §6 (la selezione trasversale) e §10 (il
confronto con chi ha fatto lo stesso lavoro) sono le tre sezioni che contano. Il
resto documenta come ci sono arrivato, comprese due strade che non portano da
nessuna parte — lo spread cross-venue e il timing temporale — che vale la pena
conoscere per non ripercorrerle.
