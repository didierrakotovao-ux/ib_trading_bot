from ibapi.client import EClient # type: ignore
from ibapi.wrapper import EWrapper # type: ignore
from ibapi.scanner import ScannerSubscription # type: ignore
import threading
import time
import xml.etree.ElementTree as ET

class ScannerApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.connectOptions = None        
        self.scanner_params_received = False
        self.all_scan_codes = []
        self.all_filters = []
        
    def scannerParameters(self, xml: str):
        """Reçoit les paramètres disponibles du scanner"""
        print("=" * 60)
        print("SCANNER PARAMETERS RECEIVED")
        print("=" * 60)
        
        # Sauvegarder le XML complet
        with open("scanner_params.xml", "w", encoding="utf-8") as f:
            f.write(xml)
        print("✓ XML sauvegardé dans 'scanner_params.xml'")
        
        # Analyser et extraire les informations
        self.parse_scanner_xml(xml)
        self.scanner_params_received = True
        
    def parse_scanner_xml(self, xml: str):
        """Parse le XML pour extraire scan codes et filtres"""
        try:
            root = ET.fromstring(xml)
            
            # 1. EXTRACT SCAN CODES (prédéfinis)
            print("\n" + "=" * 60)
            print("SCAN CODES PRÉDÉFINIS DISPONIBLES :")
            print("=" * 60)
            
            scan_codes = []
            for scan in root.findall(".//ScanCode"):
                name = scan.get('name', '')
                scan_codes.append(name)
                print(f"  • {name}")
            
            self.all_scan_codes = scan_codes
            print(f"\nTotal: {len(scan_codes)} scan codes disponibles")
            
            # 2. EXTRACT FILTERS (filtres génériques)
            print("\n" + "=" * 60)
            print("FILTRES GÉNÉRIQUES DISPONIBLES :")
            print("=" * 60)
            
            filters = []
            for filter_elem in root.findall(".//Filter"):
                filter_name = filter_elem.get('name', '')
                filter_type = filter_elem.get('type', '')
                filters.append((filter_name, filter_type))
                print(f"  • {filter_name} (type: {filter_type})")
            
            self.all_filters = filters
            print(f"\nTotal: {len(filters)} filtres disponibles")
            
            # 3. FILTRES TECHNIQUES (moyennes mobiles, indicateurs)
            print("\n" + "=" * 60)
            print("FILTRES TECHNIQUES (SMA, RSI, etc.) :")
            print("=" * 60)
            
            technical_keywords = ['SMA', 'MA', 'RSI', 'MACD', 'VOLAT', 'VOLUME', 'CHANGE', 'PRICE']
            technical_filters = []
            
            for filter_name, filter_type in filters:
                for keyword in technical_keywords:
                    if keyword in filter_name.upper():
                        technical_filters.append((filter_name, filter_type))
                        print(f"  • {filter_name} (type: {filter_type})")
                        break
            
            if not technical_filters:
                print("  Aucun filtre technique trouvé")
            
            # 4. OPTIONS DE TRI (sorting)
            print("\n" + "=" * 60)
            print("OPTIONS DE TRI DISPONIBLES :")
            print("=" * 60)
            
            sort_fields = []
            for sort in root.findall(".//Sort"):
                field = sort.get('field', '')
                sort_fields.append(field)
                print(f"  • {field}")
            
            # 5. LOCATIONS DISPONIBLES
            print("\n" + "=" * 60)
            print("LOCATIONS DISPONIBLES :")
            print("=" * 60)
            
            locations = []
            for loc in root.findall(".//Location"):
                loc_name = loc.get('name', '')
                locations.append(loc_name)
                print(f"  • {loc_name}")
            
            # 6. INSTRUMENTS DISPONIBLES
            print("\n" + "=" * 60)
            print("INSTRUMENTS DISPONIBLES :")
            print("=" * 60)
            
            instruments = []
            for instr in root.findall(".//Instrument"):
                instr_name = instr.get('name', '')
                instruments.append(instr_name)
                print(f"  • {instr_name}")
                
        except Exception as e:
            print(f"Erreur lors du parsing XML: {e}")
            # Afficher les premières lignes du XML pour debug
            print("\nPremières 500 caractères du XML :")
            print(xml[:500])

def run_loop(app):
    """Run the client loop in a separate thread"""
    app.run()

def main():
    # Configuration simple
    host = "127.0.0.1"  # Localhost
    port = 7497         # TWS paper trading : 7497, IB Gateway : 4002
    client_id = 1
    
    # Créer l'application
    app = ScannerApp()
    
    # Se connecter
    print(f"Connexion à TWS/IB Gateway sur {host}:{port}...")
    app.connect(host, port, client_id)
    
    # Démarrer le thread de communication
    api_thread = threading.Thread(target=run_loop, args=(app,), daemon=True)
    api_thread.start()
    
    # Attendre la connexion
    time.sleep(2)
    
    # Demander les paramètres du scanner
    print("\nDemande des paramètres du scanner...")
    app.reqScannerParameters()
    
    # Attendre la réponse (30 secondes max)
    print("Attente des paramètres (30 secondes max)...")
    timeout = 30
    start_time = time.time()
    
    while not app.scanner_params_received and (time.time() - start_time) < timeout:
        time.sleep(1)
        print(".", end="", flush=True)
    
    print()
    
    if app.scanner_params_received:
        print("\n✓ Paramètres reçus avec succès !")
        
        # Afficher un résumé
        print("\n" + "=" * 60)
        print("RÉSUMÉ DES DONNÉES DISPONIBLES :")
        print("=" * 60)
        print(f"Scan codes: {len(app.all_scan_codes)}")
        print(f"Filtres génériques: {len(app.all_filters)}")
        
        # Recherche spécifique des moyennes mobiles
        print("\nRecherche de filtres SMA/MA...")
        sma_filters = [f for f, t in app.all_filters if 'SMA' in f.upper() or 'MA' in f.upper()]
        if sma_filters:
            print("Filtres SMA trouvés:")
            for f in sma_filters:
                print(f"  - {f}")
        else:
            print("Aucun filtre SMA trouvé directement")
        
    else:
        print(f"\n✗ Timeout après {timeout} secondes")
        print("Vérifiez que TWS/IB Gateway est bien démarré")
    
    # Nettoyage
    time.sleep(2)
    app.disconnect()
    print("\nDéconnexion terminée.")

if __name__ == "__main__":
    main()