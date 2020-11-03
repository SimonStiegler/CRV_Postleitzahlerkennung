# CRV_Postleitzahlerkennung
Projektskizze „Erkennung von handschriftlicher Postleitzahl eines Briefes“ - Computer & Robot Vision

Projektmitglieder:
Simon Stiegler, Moritz Wiedenhöfer, Eric Bräuninger
Projektbeschreibung/Ziel: 
Für den Use Case einer automatischen Postfachzuteilung sollen Briefe erkannt und deren Adressierung analysiert werden. Hierfür soll im Rahmen dieses Projektes mit Hilfe von Methoden der Bildverarbeitung und einem neuronalen Netz die handschriftliche Postleitzahl erkannt werden. Über eine Datenbank soll die erkannte Postleitzahl anschließend einem Ort zugeordnet werden.
Programmiersprache: Python
Vorgehen: 
1.	Brief erkennen (als Objekt im Bild, Größe, Position)
2.	Bereich der Adressierung abschätzen 
(Idee: Anhand der Briefmarke die Orientierung des Briefs erkennen und auf die Position der Adresse im rechten unteren Bereich des Briefes schließen.)




3.	Segmentation einzelner Zahlen/Buchstaben im Adressbereich
4.	Erkennung von Zahlen durch neuronales Netz
5.	Ermittlung der PLZ anhand der erkannten Zahlen und deren Position
6.	Abgleich der erkannten PLZ mit einer Datenbank zur Bestimmung des Ortes
7.	Ausgabe der PLZ inklusive Ort
Projektaufteilung:
•	Bilderkennung/Segmentation: Simon, Eric
•	Neuronales Netz: Moritz
Mögliche Erweiterung:
Erkennung von Buchstaben und daraus folgend der restlichen Adressierung des Briefes 

