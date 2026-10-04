# Graf → Farkas-vittne (experimental)

Import: `from field_engine.experimental.farkas_witness import farkas_witness, verify_certificate, load_exact_json`.
Matematisk modellcertifiering; status om verklig fysiologi kräver att källans villkor är riktiga.

```python
scope = {"population": "declared model", "location": "same vessel", "window": "steady"}
model = {
    "schema": "farkas-claims/v1", "domain": "inequality", "scope": scope,
    "variables": [{"id": "x", "quantity": "normalized shear", "unit": "1"}],
    "claims": [
        {"id": "A", "variable": "x", "interval": ["0", "1"], "unit": "1",
         "scope": scope, "source": {"id": "measurement:A"}},
        {"id": "B", "variable": "x", "interval": ["3", "4"], "unit": "1",
         "scope": scope, "source": {"id": "measurement:B"}},
    ],
    "measurements": [{"id": "sensor", "quantity": "normalized shear", "coefficients": {"x": "1"},
                      "unit": "1", "where": {"location": "same vessel", "window": "steady"},
                      "scope": scope, "error_bound": "1/10", "resolution": "1/10", "cost": "1"}],
}
answer = farkas_witness(model)
assert answer["status"] == "INCONSISTENT"
assert answer["minimal_subset"] == ["A", "B"]
assert verify_certificate(model, answer["certificate"])
```

`status`: CONSISTENT kräver ett exakt primalt vittne; INCONSISTENT kräver ett exakt strikt dualt vittne; UNKNOWN anger orsak. `minimal_subset` är minsta **antal claims**, inte bara en delmängd som inte kan krympas. Certifikatet har ett nej-vittne och ett ja-vittne för varje delmängd med k−1 claims. Med monotonicitet täcker detta alla mindre delmängder. Vid budget/okända koniska subproblem kan ett nej vara verifierat medan minimum är okänt: `minimal_subset=null`, `minimality=UNKNOWN`, `conflict_subset` och nej-certifikatet finns kvar. Konsumenten måste läsa `minimum_proof`, inte tolka varje kontrollerat nej som ett bevis om minimum.

`dual_weights` listar varje använd rad med claim-ID, källa och exakt rationell vikt. En equality kan bli två LP-rader och ett intervall blir två bounds; dessa räknas fortfarande som en claim. Vikterna hör till radernas deklarerade skalning; de är inte sannolikheter eller en rangordning av källors trovärdighet. LP-variabler är fria; positivitet måste vara ett explicit villkor. `domain=orthant` betyder x≥0, `domain=soc3` betyder t≥sqrt(u²+v²) för tre koordinater med gemensam enhet efter explicit skalning. Koniska claims stöder endast likheter; blandade konprodukter saknas.

Allmän rad: `constraints=[{"coefficients":{"x":"1","y":"-1"},"relation":"<=","rhs":"2","unit":"N"}]`. Varje variabel ska ha quantity/unit. Dimensionslösa koefficienter kräver samma variabel- och radenhet. För olika enheter måste `coefficient_units` ange den explicita kvoten, t.ex. `{"area":"N/m2"}` när radens enhet är N och variabelns m2. Detta kontrollerar deklarationen, inte den fysiska lagen eller en full symbolisk enhetsalgebra. Alla claims måste ha samma explicita scope. En källa är `{"id":...}` eller `{"path":...,"sha256":...}`; filhashens sanningsenlighet är ett separat inläsningsansvar. API:t gör inga fil-/nätverksläsningar för proveniens.

Hårda tal ska vara heltal, Fraction, Decimal eller rationella/decimala strängar. Python-floats vägras. Läs ursprunglig JSON med `load_exact_json(text)` så att dess decimaltokens bevaras; använd aldrig `str(float)` för att låtsas återskapa källprecision. Intervall anger tillåtna värden på storheten. Nominala modellvärden är inte uppmätta konfidensintervall. Konverteringar stöds explicit för 1/%, M/mM/uM/µM/nM, V/mV och Pa/kPa; samma enhetssträng accepteras även för övriga storheter, övriga omräkningar ger UNKNOWN.

`certificate.decision` innehåller `kind,A,b,witness,row_labels,claim_ids`. FARKAS originalkontrollant kan kontrollera den direkt: `check_dual(d["kind"],d["A"],d["b"],d["witness"])` ger NEJ; motsvarande `check_primal` ger JA. `_farkas_check.py` är byteidentisk med förälderns `dual_witness_v1.py`; hash finns i SOURCE_MANIFEST. `verify_certificate` binder dessutom hela indatan inklusive sources/measurements och kontrollerar minimum. Fyra API-vittnen finns också som kärnkontrollerade Lean-instansbevis i `lean/ApiWitnesses.lean`; minimum, parser, OED och den fysiska bindningen är inte Lean-formaliserade.

`adapt_claim_federation(raw_graph)` använder originalformatets `graph_id/claims` med tillägg: `graph.farkas_context` är modellens variables/domain/scope/measurements och varje `claim.farkas_claim` innehåller det hårda claimkontraktet. ID blir `graph_id:claim_id`. Anropa **före** `Federation.add_graph`, som slänger okända fält och omvandlar validity till floats. Befintlig `sign` är en relationsriktning och `validity` ett tillämpningsområde; de omvandlas aldrig till ekvationer eller bound på en observerbar variabel.

`adapt_bodytwin(node)` tar ett explicit `node.farkas_model` och binder node-ID/hash; vanlig native prosa ger UNKNOWN. Det betyder inte att noden är helupplöst. `adapt_g3_pair(edge)` kräver två sidor med samma quantity/population/regime/method och vägrar REJECTED_CONVENTION. De sju äldre G3-delupplösningarna kan därför inte räknas som sju Farkas-nej.

`suggested_discriminating_measurement`: användarens tillåtna linjära sensorer prövas på modeller där en core-claim i taget tas bort. `where`, `scope`, `error_bound`, `resolution` och `cost` måste anges. Exakt projektion ger möjliga mätintervall; noise-expanderade band måste vara parvis disjunkta. Välj maximal minsta bandgap/resolution/cost bland certifierbart skiljande kandidater. Denna snäva OED betyder val av experiment under den angivna modellen. Saknad mätmodell, överlapp, obegränsade svar och konisk projektion ger UNKNOWN. Förslaget gäller antagandet att exakt en core-claim behöver tas bort; övriga claims utanför core ingår inte och ingen generell biologisk orsak eller global sensoroptimalitet hävdas.

Standardbudget: 12 claims, 16 variabler, 4096 oracle-anrop, 4096 FM-rader; minsta delmängd och exakt projektion kan ha exponentiell kostnad. LP fungerar med standardbiblioteket. Konisk upptäckt använder valfritt Clarabel + NumPy/SciPy; alla numeriska förslag kontrolleras exakt. Svag koninfeasibilitet kan sakna strikt separator och ger UNKNOWN. Återkontroll av ett sparat certifikat stödjer redigering genom en ny full inputbindning; inget inkrementellt cachelöfte ges.

Leverans: PATCH.diff mot fältmotorns a72439e23ed589dc1d49473d91688cfe4676f489. Ingen integration/commit gjord. Native pilot: 0/36 certifierade minimum, G3 tidigare 7/36 beräkningsmässiga delpar (annan endpoint). Se RESULTS.md och raw/ för kontroll, kostnad och kvarstående semantik.

Valfria sensorfel hanteras per kandidat: ogiltiga rationella tal, koefficientformat eller sensor-ID gör bara den kandidaten oanvändbar. Ett redan verifierat beslut/minimum och andra giltiga sensorer bevaras. Sensor-ID ska vara en icke-tom sträng.
