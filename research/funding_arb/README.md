# Arbitraggio sul funding rate — analisi e framework di backtest

Studio di fattibilità su dati reali: un bot delta-neutral che incassa il *funding
rate* dei perpetual, invece di scommettere sulla direzione del prezzo.

Nessun modulo qui dentro piazza ordini o usa API key. Legge dati di mercato
pubblici e produce numeri. La separazione è voluta: lo scopo di questa fase è
scoprire **se l'edge sopravvive ai costi** prima che del capitale — anche di
testnet — venga puntato addosso.

---

## Verdetto in tre righe

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
4. Il vincolo "win rate costante > 50%" **non è il vincolo giusto** e, con i
   volumi di trade che queste strategie generano, non è nemmeno verificabile
   statisticamente su meno di un anno di dati. Sotto c'è la metrica da usare al
   suo posto.

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

## 6. Rischi

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

## 7. Uso

```bash
# Report completo su una coppia
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

## 8. Cosa serve prima di passare al live

In ordine, e nessuno di questi passi è saltabile:

1. **Allungare il campione.** 96 giorni non bastano. Da Railway, rifai il fetch
   con Bybit o Binance come seconda gamba: hanno anni di storico e portano il
   campione a un livello in cui l'intervallo di Wilson comincia a dire qualcosa.
2. **Allargare l'universo.** BTC ed ETH sono gli asset con il funding più
   efficiente, cioè i peggiori per questa strategia. Il funding ricco sta sulle
   alt — dove però la liquidità è sottile e il rischio di liquidazione più alto.
   Il framework è già multi-coin: `--coins SOL AVAX DOGE ...`.
3. **Walk-forward, non backtest singolo.** Taratura dei parametri su una
   finestra, misura su quella successiva, rolling. Uno sweep ottimizzato in-sample
   come quelli del §5 è una descrizione del passato, non una previsione. Nota che
   la configurazione vincente qui — quella passiva — *non ha parametri da tarare*,
   ed è proprio per questo che è la più credibile delle due.
4. **Misurare lo slippage vero.** Metti `execution_slip_bps` a un numero
   ottenuto da fill reali su testnet, non a zero.
5. **Paper trading con esecuzione reale.** Ordini su Hyperliquid testnet che
   replicano i segnali, per misurare il ritardo fra le due gambe. È la variabile
   che il backtest non può darti.
6. **Solo allora**, capitale reale, e con la size più piccola che l'exchange
   accetta.

Un avvertimento sulla scala, visto che l'ipotesi di lavoro è testnet/paper: a
500-5.000 $ di capitale reale questa strategia **non ha senso economico**. Serve
margine su due venue contemporaneamente, i size minimi sono ~10-20 $ per ordine,
e un rendimento del 5% annuo su 2.000 $ fa 100 $ — meno del costo in tempo di
tenerla in piedi. La soglia in cui il carry cross-venue inizia a ripagare
l'infrastruttura è nell'ordine delle decine di migliaia di dollari.

---

## Struttura

```
research/funding_arb/
├── venues.py     # adapter API pubbliche + cache su disco (HL, OKX, Bybit, Binance, spot)
├── costs.py      # modello di costo e curva di breakeven
├── dataset.py    # allineamento su griglia oraria causale, normalizzazione intervalli
├── strategy.py   # EWMA causale dello spread, regole entry/exit
├── backtest.py   # motore a eventi: funding discreto, basis, liquidazione isolata
├── metrics.py    # expectancy, Wilson CI, Sharpe, drawdown, attribuzione PnL
├── run.py        # CLI
└── tests/        # 24 test, focalizzati su lookahead / normalizzazione / contabilità
```
