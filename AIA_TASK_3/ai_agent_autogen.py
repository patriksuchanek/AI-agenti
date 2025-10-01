# AI Agent pomocí Autogen s ReAct architekturou a MCP nástroji
# Lekce 7 - Praktické cvičení

import os
import json
import sqlite3
from typing import Dict, Any, List, Annotated
from dotenv import load_dotenv
from pathlib import Path
import wikipedia
from tavily import TavilyClient

# Autogen imports
from autogen import ConversableAgent, register_function
from autogen.coding import LocalCommandLineCodeExecutor

# --- Load environment variables from .env early ---
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=False)
# --------------------------------------------------

class DatabaseTool:
    """Nástroj pro práci s databází"""
    
    def __init__(self, db_path: str = "agent_database.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Inicializace databáze s ukázkovými daty"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Vytvoření tabulky uživatelů
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                age INTEGER,
                city TEXT
            )
        ''')
        
        # Vytvoření tabulky produktů
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                brand TEXT,
                model TEXT,
                price REAL,
                original_price REAL,
                category TEXT,
                subcategory TEXT,
                description TEXT,
                specifications TEXT,
                in_stock INTEGER,
                rating REAL,
                reviews_count INTEGER,
                supplier TEXT,
                added_date TEXT
            )
        ''')
        
        # Vytvoření tabulky objednávek
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                product_id INTEGER,
                quantity INTEGER,
                total_price REAL,
                order_date TEXT,
                status TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (product_id) REFERENCES products (id)
            )
        ''')
        
        # Ukázkové uživatele
        sample_users = [
            ("Jan Novák", "jan@email.com", 30, "Praha"),
            ("Marie Svobodová", "marie@email.com", 25, "Brno"),
            ("Petr Dvořák", "petr@email.com", 35, "Ostrava"),
        ]
        
        # Ukázkové produkty
        sample_products = [
            ("MacBook Air M2", "Apple", "MacBook Air 13''", 32990.0, 35990.0, "Elektronika", "Notebooky", 
             "13'' notebook s Apple M2 čipem", 
             "Apple M2 8-core CPU, 8GB RAM, 256GB SSD", 5, 4.7, 89, "iStyle", "2024-02-01"),
            
            ("ThinkPad X1 Carbon", "Lenovo", "X1 Carbon Gen 11", 45990.0, 52990.0, "Elektronika", "Notebooky", 
             "14'' business notebook", 
             "Intel Core i7, 16GB RAM, 512GB SSD", 3, 4.5, 127, "Alza", "2024-01-15"),
            
            ("iPhone 15 Pro", "Apple", "iPhone 15 Pro 128GB", 28990.0, 30990.0, "Elektronika", "Mobily", 
             "6.1'' smartphone s A17 Pro čipem", 
             "A17 Pro, 128GB, 6.1'' OLED", 8, 4.6, 203, "T-Mobile", "2024-02-10"),
            
            ("GeForce RTX 4070", "NVIDIA", "RTX 4070 SUPER", 18990.0, 21990.0, "Elektronika", "Grafické karty", 
             "Výkonná grafická karta", 
             "12GB GDDR6X, PCI Express 4.0", 6, 4.8, 94, "Mironet", "2024-01-25"),
        ]
        
        # Ukázkové objednávky
        sample_orders = [
            (1, 1, 1, 32990.0, "2024-02-25", "Doručena"),
            (2, 3, 1, 28990.0, "2024-02-26", "V přípravě"),
            (1, 4, 1, 18990.0, "2024-02-27", "Expedována"),
        ]
        
        cursor.executemany("INSERT OR IGNORE INTO users (name, email, age, city) VALUES (?, ?, ?, ?)", sample_users)
        cursor.executemany("""INSERT OR IGNORE INTO products 
                          (name, brand, model, price, original_price, category, subcategory, 
                           description, specifications, in_stock, rating, reviews_count, supplier, added_date) 
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", sample_products)
        
        cursor.executemany("""INSERT OR IGNORE INTO orders 
                          (user_id, product_id, quantity, total_price, order_date, status) 
                          VALUES (?, ?, ?, ?, ?, ?)""", sample_orders)
        
        conn.commit()
        conn.close()
    
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Vykonání SQL dotazu"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query)
            
            if query.strip().upper().startswith('SELECT'):
                results = [dict(row) for row in cursor.fetchall()]
            else:
                conn.commit()
                results = [{"affected_rows": cursor.rowcount}]
            
            conn.close()
            return results
        except Exception as e:
            return [{"error": str(e)}]


class WikipediaTool:
    """Nástroj pro vyhledávání na Wikipedii"""
    
    def search(self, query: str, lang: str = "cs") -> Dict[str, Any]:
        """Vyhledávání na Wikipedii"""
        try:
            wikipedia.set_lang(lang)
            search_results = wikipedia.search(query, results=3)
            
            if not search_results:
                return {"error": "Žádné výsledky nenalezeny"}
            
            try:
                page = wikipedia.page(search_results[0])
                summary = wikipedia.summary(search_results[0], sentences=3)
                
                return {
                    "title": page.title,
                    "summary": summary,
                    "url": page.url,
                    "related_searches": search_results[1:] if len(search_results) > 1 else []
                }
            except wikipedia.exceptions.DisambiguationError as e:
                page = wikipedia.page(e.options[0])
                summary = wikipedia.summary(e.options[0], sentences=3)
                
                return {
                    "title": page.title,
                    "summary": summary,
                    "url": page.url,
                    "disambiguation_options": e.options[:5]
                }
                
        except Exception as e:
            return {"error": f"Chyba při vyhledávání: {str(e)}"}


class TavilySearchTool:
    """Nástroj pro webové vyhledávání pomocí Tavily API"""
    
    def __init__(self):
        self.api_key = os.getenv("TAVILY_API_KEY")
        if self.api_key:
            self.client = TavilyClient(api_key=self.api_key)
        else:
            self.client = None
    
    def search(self, query: str, max_results: int = 5, include_domains: List[str] = None, 
               search_depth: str = "basic") -> Dict[str, Any]:
        """Vyhledávání pomocí Tavily API"""
        if not self.client:
            return {
                "error": "Tavily API klíč není nastaven",
                "fallback": self._fallback_search(query)
            }
        
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                include_domains=include_domains,
                search_depth=search_depth,
                include_answer=True,
                include_raw_content=False
            )
            
            return {
                "query": query,
                "answer": response.get("answer", ""),
                "results": [
                    {
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "content": result.get("content", ""),
                        "score": result.get("score", 0.0)
                    }
                    for result in response.get("results", [])
                ],
                "response_time": response.get("response_time", 0)
            }
            
        except Exception as e:
            return {
                "error": f"Chyba při vyhledávání: {str(e)}",
                "fallback": self._fallback_search(query)
            }
    
    def _fallback_search(self, query: str) -> Dict[str, Any]:
        """Fallback vyhledávání"""
        return {
            "title": f"Fallback pro: {query}",
            "content": f"Simulovaný výsledek pro: {query}. Nastavte TAVILY_API_KEY pro reálné výsledky.",
            "note": "Toto je simulované vyhledávání."
        }


# Globální instance nástrojů
db_tool = DatabaseTool()
wiki_tool = WikipediaTool()
tavily_tool = TavilySearchTool()


# Definice funkcí pro agenta
def database_query(query: Annotated[str, "SQL dotaz k vykonání"]) -> str:
    """Vykonání SQL dotazu na databázi s tabulkami users, products a orders"""
    results = db_tool.execute_query(query)
    return json.dumps(results, ensure_ascii=False, indent=2)


def wikipedia_search(
    query: Annotated[str, "Vyhledávací dotaz"], 
    lang: Annotated[str, "Jazyk (cs/en)"] = "cs"
) -> str:
    """Vyhledávání na Wikipedii"""
    result = wiki_tool.search(query, lang)
    return json.dumps(result, ensure_ascii=False, indent=2)


def web_search(
    query: Annotated[str, "Vyhledávací dotaz"],
    max_results: Annotated[int, "Počet výsledků"] = 5
) -> str:
    """Webové vyhledávání pomocí Tavily API"""
    result = tavily_tool.search(query, max_results)
    return json.dumps(result, ensure_ascii=False, indent=2)


def product_search(
    query: Annotated[str, "Název produktu"],
    category: Annotated[str, "Kategorie"] = None
) -> str:
    """Vyhledávání produktů na e-shopech"""
    search_query = f"{query} cena recenze"
    if category:
        search_query += f" {category}"
    
    domains = ["heureka.cz", "zbozi.cz", "alza.cz"]
    result = tavily_tool.search(search_query, max_results=3, include_domains=domains, search_depth="advanced")
    return json.dumps(result, ensure_ascii=False, indent=2)


def price_comparison(
    product_name: Annotated[str, "Název produktu"],
    brand: Annotated[str, "Značka"] = None
) -> str:
    """Srovnání cen produktu"""
    query = f"{brand} {product_name}" if brand else product_name
    query += " cena srovnání kde koupit"
    
    result = tavily_tool.search(
        query=query,
        max_results=5,
        include_domains=["heureka.cz", "zbozi.cz", "pricemania.cz"],
        search_depth="advanced"
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


def create_react_agent():
    """Vytvoření ReAct agenta"""
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    llm_config = {
        "config_list": [
            {
                "model": "gpt-4",
                "api_key": openai_api_key,
                "temperature": 0.1,
            }
        ]
    }
    
    system_message = """
Jsi inteligentní asistent s přístupem k různým nástrojům. Používáš ReAct (Reasoning + Acting) metodologii:

1. **Thought** - Uvažuj o problému a naplánuj postup
2. **Action** - Vyber a použij vhodný nástroj
3. **Observation** - Vyhodnoť výsledek
4. **Thought** - Znovu uvažuj na základě získaných informací
5. Opakuj dokud nemáš kompletní odpověď

DOSTUPNÉ NÁSTROJE:

1. **database_query(query: str)** - SQL dotaz na databázi
   Tabulky: users, products, orders
   
2. **wikipedia_search(query: str, lang: str = 'cs')** - Vyhledávání na Wikipedii

3. **web_search(query: str, max_results: int = 5)** - Webové vyhledávání

4. **product_search(query: str, category: str = None)** - Vyhledávání produktů

5. **price_comparison(product_name: str, brand: str = None)** - Srovnání cen

Vždy odpovídej v češtině a postupuj podle ReAct schématu.
"""
    
    # Vytvoření agenta
    agent = ConversableAgent(
        name="ReactAgent",
        system_message=system_message,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )
    
    # Vytvoření user proxy
    user_proxy = ConversableAgent(
        name="UserProxy",
        llm_config=False,
        is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
        human_input_mode="NEVER",
    )
    
    # Registrace funkcí
    register_function(
        database_query,
        caller=agent,
        executor=user_proxy,
        name="database_query",
        description="Vykonání SQL dotazu na databázi"
    )
    
    register_function(
        wikipedia_search,
        caller=agent,
        executor=user_proxy,
        name="wikipedia_search",
        description="Vyhledávání na Wikipedii"
    )
    
    register_function(
        web_search,
        caller=agent,
        executor=user_proxy,
        name="web_search",
        description="Webové vyhledávání"
    )
    
    register_function(
        product_search,
        caller=agent,
        executor=user_proxy,
        name="product_search",
        description="Vyhledávání produktů"
    )
    
    register_function(
        price_comparison,
        caller=agent,
        executor=user_proxy,
        name="price_comparison",
        description="Srovnání cen"
    )
    
    return agent, user_proxy


def process_query(agent, user_proxy, query: str) -> str:
    """Zpracování dotazu"""
    try:
        result = user_proxy.initiate_chat(
            agent,
            message=f"Prosím odpověz na tento dotaz pomocí ReAct metodologie: {query}",
            max_turns=15,
            silent=False
        )
        
        # Získání poslední zprávy
        messages = user_proxy.chat_messages.get(agent, [])
        if messages:
            last_msg = messages[-1].get("content", "")
            return last_msg
        else:
            return "Nepodařilo se získat odpověď"
            
    except Exception as e:
        return f"Chyba: {str(e)}"


def main():
    """Hlavní funkce"""
    print("🤖 AI Agent s ReAct architekturou a MCP nástroji")
    print("=" * 60)
    
    # Vytvoření agenta
    agent, user_proxy = create_react_agent()
    
    # Testovací dotazy
    test_queries = [
        "Najdi všechny notebooky značky Apple v databázi",
        "Kolik produktů máme v kategorii Elektronika?",
        "Zobraz objednávky uživatele Jana Nováka",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {query}")
        print("-" * 50)
        
        try:
            response = process_query(agent, user_proxy, query)
            print(f"\n✅ Odpověď:\n{response}")
        except Exception as e:
            print(f"❌ Chyba: {e}")
        
        print("\n" + "="*60)


if __name__ == "__main__":
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if not openai_key:
        print("⚠️  Nastavte OPENAI_API_KEY v .env souboru")
        exit(1)
    
    if not tavily_key:
        print("⚠️  TAVILY_API_KEY není nastaven - použije se fallback vyhledávání")
    
    main()