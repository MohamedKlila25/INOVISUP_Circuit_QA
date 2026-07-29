# CircuitVQA v4 — Format d'annotations & workflow

**48 templates · 22 classes · 2 domaines (électrique / logique)**

## Installation

```bash
cd notebooks/
pip install schemdraw matplotlib pillow opencv-python tqdm
```

> `cairosvg` n'est plus nécessaire depuis la v4 : le PNG est rasterisé
> directement depuis la figure matplotlib (voir « Précision » plus bas).

## Génération

```bash
# Script (recommandé sur cluster)
python data_generation/generate_dataset.py --out data/circuitvqa_v4

# Contrôle qualité automatique
python data_generation/validate_dataset.py --dataset data/circuitvqa_v4
```

Ou via le notebook `CircuitVQA_v4_pipeline.ipynb` (mêmes fonctions
importées depuis `data_generation/`, donc résultat identique).

## Structure de sortie

```
data/circuitvqa_v4/
├── images/{train,test}/          circuit_XXXXX.png  (800 px de large)
├── labels/{train,test}/          circuit_XXXXX.txt  (class xc yc w h — nom imposé par ultralytics)
├── annotations/{train,test}/     circuit_XXXXX.json (annotation complète)
├── circuit.yaml                  config YOLO prête à l'emploi
└── summary.json                  statistiques par template
```

## Format `annotations/*.json`

```json
{
  "image": "images/train/circuit_00000.png",
  "image_size": [800, 664],
  "template": "zener_regulator",
  "domain": "electrical",
  "circuit": {
    "components": [{"id": "R1", "class": "resistor", "value": "4.7kΩ"}],
    "connections": [{"from": "V1", "to": "R1"}],
    "circuit_metadata": {"topology": "mixed", "domain": "electrical"}
  },
  "components": [
    {
      "id": "R1", "class": "resistor", "class_idx": 0,
      "bbox": [x0, y0, x1, y1],
      "terminals": [
        {"name": "start", "x": 0, "y": 0},
        {"name": "end",   "x": 0, "y": 0}
      ]
    }
  ],
  "texts": [
    {
      "comp_id": "R1",
      "text": "R1 4.7kΩ",
      "id_text": "R1",
      "value_text": "4.7kΩ",
      "bbox": [x0, y0, x1, y1]
    }
  ],
  "nets": [
    {"net_id": 0, "terminals": [
      {"comp_id": "V1", "terminal": "end"},
      {"comp_id": "R1", "terminal": "start"}]}
  ],
  "junctions": [{"x": 0, "y": 0, "degree": 3, "type": "junction"}],
  "crossovers": []
}
```

- `connections` : sens = polarité (`from` = pôle +/anode).
- `nets` : **vérité terrain des connexions**. Deux terminaux partageant
  un `net_id` sont reliés par un fil. Toutes les masses forment un seul
  net global (convention netlist).
- `crossovers` : vide — le catalogue est 100 % planaire, aucun
  croisement sans connexion. Le champ existe pour du code prêt à
  l'emploi si de futurs templates en introduisent.

### Nombre de terminaux selon la classe

| Classe | Terminaux |
|---|---|
| composants 2 broches (R, C, L, diodes, sources, interrupteur, fusible) | `start`, `end` |
| AOP | `in1`, `in2`, `out` |
| transistors NPN / PNP | `base`, `collector`, `emitter` |
| portes logiques | `in1`…`inN`, `out` |

## Les deux domaines

Les circuits **logiques** et **électriques** ne sont jamais mélangés :
un schéma logique ne contient que des portes (entrées A/B/C à gauche,
sortie Y à droite). Conséquence : un schéma logique est **ouvert**, donc
le contrôle de fermeture de boucle ne s'y applique pas.

Répartition par défaut : 75 % électrique / 25 % logique
(`DOMAIN_WEIGHTS` dans `generate_dataset.py`).

## Garanties de qualité (vérifiées automatiquement)

1. **Boucle fermée** (domaine électrique) : flood fill — le circuit
   enclôt au moins une région de fond.
2. **Bbox composant** : contient le symbole entier (zigzag complet des
   résistances), fils de connexion exclus.
3. **Terminaux** : chaque point tombe sur un fil.
4. **Bbox texte** : contient le texte rendu.
5. **Jonctions** : sur un fil, degré 3 ou 4 (les simples coins exclus).
6. **Nets** : chaque connexion du catalogue partage un net, et aucun
   court-circuit (jamais les deux bornes d'un composant dans le même net).
7. **Lisibilité** : aucune étiquette ne recouvre une autre étiquette ni
   le symbole d'un autre composant. Un schéma peut être électriquement
   juste et parfaitement annoté tout en étant graphiquement illisible —
   les contrôles 1 à 6 laissent passer ce cas, celui-ci l'attrape.

Score de référence : **960 circuits → 100 % de qualité (les 7 contrôles),
100 % d'images visuellement uniques, aucune classe vide**.

## Diversité

Chaque image tire un style au hasard : résistances IEEE ou IEC,
épaisseur de trait, taille de police, police, échelle, espacement,
position des labels. L'ordre des composants des boucles série est
mélangé (les connexions sont reconstruites en conséquence). Un quota
`RARE_SHARE` réservé **par domaine** garantit assez d'exemples pour les
classes rares (transistors, XNOR, fusible…).

## Précision des coordonnées

Le PNG est rasterisé **directement depuis la figure matplotlib**, et non
via un SVG intermédiaire : schemdraw écrit ses SVG en « bbox tight »,
donc leur taille diffère de celle de la figure dès qu'une étiquette
déborde. L'échelle déduite des dimensions de la figure serait alors
fausse d'environ 1 %, et les terminaux tomberaient à côté des fils sur
les images larges. Écart terminal mesuré après correction : **0,00 px**.

## Étapes suivantes

1. **Détection** : `yolo detect train data=data/circuitvqa_v4/circuit.yaml model=yolo11n.pt epochs=100 imgsz=800`
2. **Régression de terminaux** : supervision sur `components[i].terminals`
   (nombre variable selon la classe, voir tableau ci-dessus).
3. **Pathfinding** : BFS sur le masque de fils entre terminaux prédits ;
   une paire prédite est correcte si les deux terminaux partagent un
   `net_id`. Les `junctions` sont traversables.
4. **OCR** : crops `texts[i].bbox` → vérité `id_text` / `value_text`,
   appariement au composant par `comp_id`.

## Fichiers du pipeline

| Fichier | Rôle |
|---|---|
| `circuit_catalog.py` | 48 templates fermés, valeurs E12, deux domaines |
| `renderer.py` | Rendu générique série/parallèle/pont/mixte + styles aléatoires |
| `custom_renders.py` | Rendus dédiés : AOP, transistors, pont de Graetz, redresseur |
| `logic_renders.py` | Rendus des circuits logiques (portes seules) |
| `renderer_annotated.py` | Annotation : bbox, terminaux, textes, jonctions, nets |
| `generate_dataset.py` | Orchestration, labels YOLO, YAML, équilibrage des classes |
| `validate_dataset.py` | Contrôle qualité automatique post-génération |
