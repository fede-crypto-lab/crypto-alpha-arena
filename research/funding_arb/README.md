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
vale **1-3% APR sul capitale**, e la sua viabilità dipende interamente da un
parametro che i dati storici non possono darmi — lo slippage reale di esecuzione
sulle alt. Va misurato su testnet prima di qualsiasi altra cosa.

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

## 8. Rischi

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

## 9. Uso

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

## 10. Runbook operativo

### Quello che NON serve

**TAAPI.** Il repo ha un client con `backtrack`/`results` per scaricare indicatori
storici (EMA, MACD, RSI). Per questa strategia sono **irrilevanti**, e usarli
sarebbe tornare all'approccio previsivo che è già fallito al §5: l'edge qui è un
premio strutturale, pagato a chi fornisce il lato impopolare del book, non un
pattern che un oscillatore possa anticipare. Il piano gratuito è anche limitato a
1 richiesta ogni 15 secondi, quindi un download storico di massa non è comunque
praticabile.

**La testnet, per validare l'edge.** Un book di testnet è una manciata di ordini
sintetici: ci misureresti la testnet, non il mercato. Lo slippage — l'unico numero
che mancava — si legge dal book *pubblico* di mainnet, gratis e senza rischio
(§7). La testnet serve a una cosa sola, ed è il punto 4 qui sotto.

### Quello che serve, in ordine

**1. Accumulare la distribuzione dello slippage.** È l'unico passo che richiede
tempo di calendario, e quindi l'unico che devi far partire tu. Uno snapshot
singolo non basta: la profondità alle 3 di notte di domenica e quella durante una
cascata di liquidazioni differiscono di un ordine di grandezza — e un carry va
chiuso proprio nel secondo tipo di momento.

```bash
python -m research.funding_arb.run --liquidity --notional 10000 \
    --universe-size 30 --sample-to data/liquidity.jsonl
```

Mettilo in cron **ogni ora** su Railway. Il file è JSON Lines, in append: ogni
riga è uno snapshot per coin. Dopo 2-4 settimane hai una distribuzione, e il
numero contro cui pianificare è un **percentile cattivo** (p90, p95), non la
mediana. Rilanciando il comando ti stampa anche mediana e caso peggiore per coin.

**2. Rifare il backtest con i costi veri.** Quando il file ha abbastanza
campioni:

```bash
python -m research.funding_arb.run --portfolio --days 180 \
    --venue-a okx_spot --venue-b hyperliquid --universe-size 24 \
    --max-positions 5 --entry-rank 5 --exit-rank 20 --rank-lookback 168 \
    --min-hold 480 --leverage 1 \
    --use-measured-slippage --max-slippage-bps 40
```

Se l'APR regge con il p90 al posto dello snapshot istantaneo, la strategia esiste.
Se non regge, hai risparmiato il capitale.

**3. Allungare il campione storico.** 180 giorni in una finestra rialzista sono
pochi, e il funding è strutturalmente più ricco in bull market. Da Railway, Bybit
e Binance sono raggiungibili (da altri host CloudFront li blocca) e hanno anni di
storico: rifai il §6 con `--venue-b bybit` per vedere se ρ ≈ 0.65 regge anche in
un ribasso.

**4. Solo adesso la testnet, e solo per l'impianto.** Non per validare l'edge, ma
per verificare che la meccanica funzioni end-to-end: che le due gambe partano
insieme, che gli arrotondamenti di size passino, che il codice si accorga di un
fill parziale. `HYPERLIQUID_NETWORK=testnet` è già supportato in
`config_loader.py`. La domanda a cui rispondere è una sola: **quanti secondi
passano fra il fill della prima gamba e quello della seconda?** In quel buco sei
direzionale, ed è l'unica variabile che né il book pubblico né i dati storici
possono darti. Misurala e mettila in `CostModel.execution_slip_bps`.

**5. Walk-forward prima del capitale reale.** Le 144 configurazioni del §6 sono
valutate in-sample. Taratura su una finestra, misura su quella successiva, rolling.

**6. Poi, e solo poi**, capitale reale alla size minima.

### Un avvertimento sulla scala

A 500-5.000 $ questa strategia **non ha senso economico**. Serve margine su due
venue contemporaneamente, i size minimi sono ~10-20 $ per ordine, e il 2% annuo su
2.000 $ fa 40 $. Peggio: la tabella del §7 misura lo slippage a 10.000 $ per
gamba; a size più piccole paghi comunque il mezzo spread, che su una alt sottile
può essere più largo dell'intero rendimento annuo. La soglia in cui il carry
ripaga l'infrastruttura è nell'ordine delle decine di migliaia di dollari.

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
├── metrics.py     # expectancy, Wilson CI, Sharpe, drawdown, attribuzione PnL
├── run.py         # CLI (4 modalità: single-pair, --persistence, --portfolio,
│                  #      --liquidity)
└── tests/         # 45 test: lookahead, normalizzazione, contabilità del funding,
                   # pareggi nel ranking, aritmetica del book
```

## Ordine di lettura consigliato

§3 (l'aritmetica del breakeven) e §6 (la selezione trasversale) sono le due
sezioni che contano. Il resto documenta come ci sono arrivato, comprese due
strade che non portano da nessuna parte — lo spread cross-venue e il timing
temporale — che vale la pena conoscere per non ripercorrerle.
