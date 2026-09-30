# Provenance of the feature-space layer

The pre-registration in `docs/preregistration/` and the JSONs in `docs/v4/` were produced in the authors' source repository, whose history the anonymous mirror flattens. This file records the commits of that repository the paper relies on, as hash and UTC date, with their messages copied verbatim (they were written in Portuguese). Hashes identify commits, not people.

## Milestones of Section 3.3

| commit | date (UTC) | what | message |
|---|---|---|---|
| `531cb78` | 2026-09-17T07:17:17+00:00 | pre-registration of the feature-space analyses, before any embedding | v4: pre-registro da analise em espaco de caracteristicas, antes de qualquer embedding |
| `3e7c63e` | 2026-09-17T07:18:30+00:00 | amendment 7.1 | v4: emenda 7.1 ao pre-registro -- os rotulos que o G0 pode de fato usar, antes de rodar |
| `5e5231b` | 2026-09-17T07:27:09+00:00 | amendment 7.2 | v4: emenda 7.2 -- o G0 passa a testar conteudo, nao rotulo; piso de cor. Antes de rodar |
| `47af0be` | 2026-09-17T19:52:31+00:00 | gate G1 and analysis A; amendment 7.9 | v4: G1 PASSA -- analise A (160 celulas) e a emenda 7.9 que delimita o que e criterio |

The pre-registration (`531cb78`, 07:17 UTC on 17 September 2026) precedes the first embedding run and the first gate (`47af0be`, 19:52 UTC the same day); the amendments between them are dated in the file itself (Section 7 of `preregistration_v4.md`).

## Commits stamped in the numbers sheet

`docs/v4/numbers_v4.md` cites, under every number, the JSON it is read from and the commit of the source repository that last changed that JSON. Those commits, in date order:

| commit | date (UTC) | message |
|---|---|---|
| `9ce13ee` | 2026-09-17T14:37:29+00:00 | v4: legendas por diretorio do caso; a leitura de cada base registrada em suas proprias palavras |
| `8aad98c` | 2026-09-17T15:43:45+00:00 | v4: completo x estrito -- a sobreposicao de conteudo existe e nao move nenhum numero |
| `be515b8` | 2026-09-17T17:18:00+00:00 | v4: a oclusao colapsa entradas distintas; o ruido degrada sem colapsar |
| `1b6eade` | 2026-09-17T17:31:23+00:00 | v4: distancia ao 1-vizinho no banco -- a auditoria de duplicatas por um segundo instrumento |
| `47af0be` | 2026-09-17T19:52:31+00:00 | v4: G1 PASSA -- analise A (160 celulas) e a emenda 7.9 que delimita o que e criterio |
| `6134d57` | 2026-09-17T20:46:46+00:00 | v4: inventario de entrada da B -- verificacao 7 passa em SLAKE e VQA-RAD e falha no Kvasir |
| `affb321` | 2026-09-17T20:53:03+00:00 | v4: analise B no SLAKE e no VQA-RAD -- as duas separam, e a cor separa junto |
| `7a43c6e` | 2026-09-17T20:56:55+00:00 | v4: B no Kvasir sobre os 478 pares declarados, e o vies da exclusao medido |
| `1c187c6` | 2026-09-17T21:59:31+00:00 | v4: G2 PASSA no fio -- 3 de 4 separam, HEAL unimodal, e a ponte para o phi da v3 |
| `8f28260` | 2026-09-17T22:58:32+00:00 | v4: d por regua x base, com ranking dentro de cada regua -- e a leitura anterior corrigida |
| `1546fb0` | 2026-09-18T00:03:57+00:00 | v4: recalculo das quantidades da v3 a partir dos manifestos -- o phi do C5 muda na primeira casa |
| `ef10c9d` | 2026-09-18T00:21:24+00:00 | v4: Eq. 3 recalculada dos manifestos -- a identidade fecha, 1,81/1,72 nao mudam, d_N muda |
| `7003769` | 2026-09-18T01:13:01+00:00 | v4: G3 PASSA, 14 de 15 -- e o controle dentro do bucket inverte o quadro |
| `7a7a093` | 2026-09-18T01:34:47+00:00 | v4: onde os pares de C5 e C6 caem no eixo de distancia -- a frase do item 4, corrigida |
| `d84b4f1` | 2026-09-18T01:47:55+00:00 | v4: a_I - k por template -- a diferenca NAO e composicao de templates |
| `7c4d128` | 2026-09-18T01:58:08+00:00 | v4: D com suporte por bin e as duas ponderacoes -- a inversao NAO se sustenta pela regra |
| `91b9a0c` | 2026-09-18T02:48:01+00:00 | v4: vistas x nao vistas (secao 5) -- a comparacao e estruturalmente impossivel, e isso e o achado |
| `e0ee640` | 2026-09-18T10:27:46+00:00 | v4: E1 -- patch shuffle (C7) como condicao so de embedding, e as duas previsoes pontuadas |
| `81469aa` | 2026-09-18T12:12:15+00:00 | E2 no PathVQA: emenda E.1, e o contraste que ela previa |
| `323ba67` | 2026-09-20T03:47:13+00:00 | fig3: eixo y em VDD (pp), nao em queda relativa |
| `6c9a867` | 2026-09-20T04:41:43+00:00 | fig5: 1x3, um painel por regua e uma curva por benchmark |
| `904c6d5` | 2026-09-20T14:23:29+00:00 | v4: |V| C6 SLAKE, destino da resposta alterada, tercis em pp, C1 em fechadas, legenda da fig5, limpeza das tabelas |
| `c4b6762` | 2026-09-20T23:43:33+00:00 | v4: MedVLThinker re-pontuado pela E7 na Tabela 3; identidade CCR 30/30 e as duas fontes de nao-determinismo |
| `62686ea` | 2026-09-21T00:33:12+00:00 | v4: as tres pendencias medidas -- |I| sob os dois mapas de doadora, recontagem dos arquivos de predicao, definicao das 30 celulas |
| `3542fbc` | 2026-09-21T00:42:09+00:00 | v4: manifesto corrigido adotado na CCR, os cinco tokens em saida afinada, as 30 celulas, e a causa da identidade nao fechar no zero-shot |
| `922b3b6` | 2026-09-29T23:58:07-03:00 | v4: E7 applied to the zero-shot Kvasir C6 file; Table 3 cell 7.3 -> 4.2; rulers on the training bank stamped in the sheet |

Inside this repository the sheet is regenerated by `src/featurespace/make_numbers.py`, which stamps each line with the commit of this repository instead; the table above is the bridge between the two histories.
