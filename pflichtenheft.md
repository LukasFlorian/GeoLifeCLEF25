# Pflichtenheft

**Gruppe:**  
Yannik Angeli, Nils Kubousek, Lukas Richter, Diego Rubio Carrera, Maximilian Seible

**Projekt:**  
Speziesverteilungsmodell für die GeoLifeCLEF25-Challenge (GLC25)



## Allgemeines
Im Rahmen der GLC25 soll ein Modell zur Vorhersage der Speziesverteilung in einem Gebiet entwickelt werden.

### Ausgangssituation
Die GeoLifeCLEF 2025 Challenge zielt darauf ab, die Präsenz von Pflanzenarten an spezifischen geografischen Standorten vorherzusagen. Dies unterstützt die Biodiversitätsforschung, das Management von Ökosystemen sowie die Entwicklung von Werkzeugen zur Artenidentifikation.

### Datensatz
Der Datensatz besteht aus Satellitenbildern, Klimazeitreihen und Umweltrastern mit Indikatoren zum menschlichen Fußabdruck. Die Datenpunkte sind gelabelt mit der Präsenz oder Abwesenheit von 11255 verschiedenen Pflanzenarten.

Konkreter ist der Datensatz gegliedert in:
- Presence-Absence (PA) Daten:
  - Hier garantiert das Label 1 die Anwesenheit einer Art
  - Das Label Null bedeutet, dass die Art nicht an diesem Ort vorkommt
  - Der Datensatz enthält etwa 100.000 solcher Datenpunkte
- Presence-Only (PO) Daten:
  - Hier garantiert das Label 1 weiterhin die Anwesenheit einer Art
  - Das Label 0 impliziert hingegen nicht mehr die Abwesenheit der Spezies, sondern nur, dass sie nicht als anwesend erfasst wurde
  - Hiervon gibt es rund 5 Millionen Datenpunkte

### Projektbezug
Ziel des Projektes vor dem Hintergrund der Kaggle-Challenge ist:
- in erster Linie ein Modell zur Vorhersage der Artenpräsenz in einem bestimmten Gebiet auf Grundlage der Datenpunkte
- optional eine statistische Betrachtung der Speziesverteilung in den Datenpunkten
- optional eine Visualisierung der Datenpunkte auf einer Karte



## Funktionale Anforderungen
### Notwendige Anforderungen
- Ein Dataloader muss die Datenpunkte einlesen und in ein Format bringen, das von einem Modell verarbeitet werden kann
- Ein muss ein Modell zur Vorhersage der Artenpräsenz in einem bestimmten Gebiet auf Grundlage der Datenpunkte trainiert werden
- Das Modell muss sowohl mit PA als auch PO Daten trainiert sein
- Das Modell muss mit Hilfe des Micro F1-Scores bewertet werden
- Ein konkreter Score muss nicht erreicht werden
- Das Modell muss mindestens die Satellitenbilder als Input verwenden
- Hierfür muss ein CNN verwendet werden

### Optionale Anforderungen
- Die Speziesverteilung in den Datenpunkten soll statistisch betrachtet werden
- Die Datenpunkte sollen auf einer Karte visualisiert werden
- Das Modell soll auch die Klimazeitreihen zu einem Datenpunkt als Input verwenden
  - Hierfür soll ein MLP verwendet werden
- Das Modell soll auch die Umweltfaktoren zu einem Datenpunkt als Input verwenden
  - Auch hierfür soll ein MLP verwendet werden
- Zu dem Modell soll ein Dashboard erstellt werden, dass die Ergebnisse visualisiert
- Das Dashboard soll Nutzern ermöglichen, eigene Datenpunkte hinzuzufügen und die Vorhersage des Modells zu erhalten
- Die Nutzer-eigenen Datenpunkte sollen in einer Datenbank gespeichert werden und im Dashboard abrufbar sein
- Der Nutzer soll die Möglichkeit haben, durch Bereitstellung eigener PA-Labels zu den Eingabedaten das Modell weiter mit eigenen Datenpunkten zu trainieren



## Nichtfunktionale Anforderungen
### Notwendige Anforderungen
- Das Modell muss Gebrauch von den angelernten Features eines anderen Modells mittels Transfer Learning machen
- Das Modell muss in Python implementiert werden
- Konkreter muss die PyTorch-Library verwendet werden

### Optionale Anforderungen
- Das Modell soll mittels Parallelisierung beschleunigt werden
- Die Parallelisierung soll mit MPS und CUDA unterstützt werden
- Das Modell soll in einem Docker-Container bereitgestellt werden
- Das Modell soll mit dem Leaderboard der GLC25-Challenge evaluiert werden
- Das Training soll den 



## Rahmenbedingungen
### Technische Grundvoraussetzungen
- Das Modell soll sich lokal auf einem Rechner mit 48GB Arbeitsspeicher und ohne dezidierte Grafikkarte trainieren lassen



## Modularisierung und Einteilung der Arbeitspakete

### Modularisierung
- Das Modell soll in mehrere Module unterteilt werden
- Die Module sollen die folgenden Funktionen haben:
  - Datenvorbereitung (Preprocessing)
  - Modelldefinition
  - Modellimplementierung
  - Training Loop
  - Evaluierung
  - Containerisierung
  - Dashboard mit Datenbank der Datenpunkte