# Pflichtenheft

**Gruppe:**
Yannik Angeli, Nils Kubousek, Lukas Richter, Diego Rubio Carrera, Maximilian Seible

**Projekt:**
Speziesverteilungsmodell für die GeoLifeCLEF25-Challenge (GLC25)

## 1. Einführung

### 1.1 Allgemeines
Im Rahmen der GLC25 soll ein Modell zur Vorhersage der Speziesverteilung in einem Gebiet entwickelt werden.

### 1.2 Ausgangssituation
Die GeoLifeCLEF 2025 Challenge zielt darauf ab, die Präsenz von Pflanzenarten an spezifischen geografischen Standorten vorherzusagen. Dies unterstützt die Biodiversitätsforschung, das Management von Ökosystemen sowie die Entwicklung von Werkzeugen zur Artenidentifikation.

### 1.3 Datensatz
Der Datensatz besteht aus Satellitenbildern, Klimazeitreihen und Umweltrastern mit Indikatoren zum menschlichen Fußabdruck. Die Datenpunkte sind gelabelt mit der Präsenz oder Abwesenheit von 11255 verschiedenen Pflanzenarten.

| Datentyp | Beschreibung | Anzahl der Datenpunkte |
|----------|-------------|-----------------------|
| Presence-Absence (PA) | Label 1 garantiert die Anwesenheit einer Art, Label 0 bedeutet Abwesenheit | ca. 100.000 |
| Presence-Only (PO) | Label 1 garantiert die Anwesenheit einer Art, Label 0 bedeutet, dass die Art nicht als anwesend erfasst wurde | ca. 5 Millionen |

### 1.4 Projektziele
| Ziel | Beschreibung |
|------|-------------|
| Primäres Ziel | Entwicklung eines Modells zur Vorhersage der Artenpräsenz in einem bestimmten Gebiet auf Grundlage der Datenpunkte |
| Optionales Ziel | Statistische Betrachtung der Speziesverteilung in den Datenpunkten |
| Optionales Ziel | Visualisierung der Datenpunkte auf einer Karte |

## 2. Funktionale Anforderungen

### 2.1 Notwendige Anforderungen
| ID | Anforderung |
|----|------------|
| F-001 | Ein Dataloader muss die Datenpunkte einlesen und in ein Format bringen, das von einem Modell verarbeitet werden kann. |
| F-002 | Ein Modell zur Vorhersage der Artenpräsenz in einem bestimmten Gebiet muss auf Grundlage der Datenpunkte trainiert werden. |
| F-003 | Das Modell muss sowohl mit PA als auch PO Daten trainiert werden. |
| F-004 | Das Modell muss mit Hilfe des Micro F1-Scores bewertet werden - auch ohne Implementierung via Kaggle möglich. |
| F-005 | Das Modell muss mindestens die Satellitenbilder als Input verwenden. |
| F-006 | Für die Verwendung von Satellitenbildern muss ein CNN verwendet werden. |

### 2.2 Optionale Anforderungen
| ID | Anforderung |
|----|------------|
| F-007 | Die Speziesverteilung in den Datenpunkten soll statistisch betrachtet werden. |
| F-008 | Die Datenpunkte sollen auf einer Karte visualisiert werden. |
| F-009 | Das Modell soll auch die Klimazeitreihen zu einem Datenpunkt als Input verwenden. |
| F-010 | Für die Verwendung von Klimazeitreihen soll ein MLP verwendet werden. |
| F-011 | Das Modell soll auch die Umweltfaktoren zu einem Datenpunkt als Input verwenden. |
| F-012 | Für die Verwendung von Umweltfaktoren soll ein MLP verwendet werden. |
| F-013 | Zu dem Modell soll ein Dashboard erstellt werden, das die Ergebnisse visualisiert. |
| F-014 | Das Dashboard soll Nutzern ermöglichen, eigene Datenpunkte hinzuzufügen und die Vorhersage des Modells zu erhalten. |
| F-015 | Die Nutzer-eigenen Datenpunkte sollen in einer Datenbank gespeichert werden und im Dashboard abrufbar sein. |
| F-016 | Der Nutzer soll die Möglichkeit haben, durch Bereitstellung eigener PA-Labels zu den Eingabedaten das Modell weiter mit eigenen Datenpunkten zu trainieren. |

## 3. Nichtfunktionale Anforderungen

### 3.1 Notwendige Anforderungen
| ID | Anforderung |
|----|------------|
| NF-001 | Das Modell muss Gebrauch von den angelernten Features eines anderen Modells mittels Transfer Learning machen. |
| NF-002 | Das Modell muss in Python implementiert werden. |
| NF-003 | Es muss die PyTorch-Library verwendet werden. |

### 3.2 Optionale Anforderungen
| ID | Anforderung |
|----|------------|
| NF-004 | Das Modell soll mittels Parallelisierung beschleunigt werden. |
| NF-005 | Die Parallelisierung soll mit MPS und CUDA unterstützt werden. |
| NF-006 | Das Modell soll in einem Docker-Container bereitgestellt werden. |
| NF-007 | Das Modell soll mit dem Leaderboard der GLC25-Challenge evaluiert werden. |

## 4. Rahmenbedingungen

### 4.1 Technische Grundvoraussetzungen
| ID | Anforderung |
|----|------------|
| R-001 | Das Modell soll sich lokal auf einem Rechner mit 48GB Arbeitsspeicher und ohne dezidierte Grafikkarte trainieren lassen. |

## 5. Modularisierung und Einteilung der Arbeitspakete

### 5.1 Modularisierung
| Modul | Beschreibung |
|-------|-------------|
| Datenvorbereitung | Preprocessing der Daten |
| Modelldefinition | Definition des Modells |
| Modellimplementierung | Implementierung des Modells |
| Training Loop | Training des Modells |
| Evaluierung | Evaluierung des Modells |
| Containerisierung | Bereitstellung des Modells in einem Docker-Container |
| Dashboard | Erstellung eines Dashboards zur Visualisierung der Ergebnisse und Interaktion mit Nutzern |
