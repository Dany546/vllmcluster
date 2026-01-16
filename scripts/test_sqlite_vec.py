#!/usr/bin/env python3
"""Test script for sqlite-vec registration and basic vec0 virtual table operations.

Usage: run in the `dino` venv where you built local sqlite and vec0.

The script will try multiple registration methods in order:
 1) APSW loadextension (PRIORITY)
 2) sqlite_vec.sqlite3 wrapper (if extension loading is supported)
 3) stdlib sqlite3 with extension APIs
 4) ctypes loader against local libsqlite3

It then attempts to create a small DB, a virtual table using `vec0`, insert one vector,
and run a simple query.
"""
import os
import sys
import traceback
from pathlib import Path

DB = Path("test_vec0_auto.db")
VEC_SO_ENV = os.environ.get("VECTOR_EXT_PATH") or os.environ.get("VEC0_SO")

# Add common paths to sys.path for better module discovery
sys.path.insert(0, str(Path.home() / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"))

def find_vec_so():
    if VEC_SO_ENV:
        if os.path.exists(VEC_SO_ENV):
            return VEC_SO_ENV
        else:
            print(f"Warning: VEC0_SO environment variable points to non-existent file: {VEC_SO_ENV}")
    
    # Try sqlite_vec module path
    try:
        import sqlite_vec
        try:
            p = sqlite_vec.loadable_path()
            if p and os.path.exists(p):
                print(f"Found vec0.so via sqlite_vec.loadable_path(): {p}")
                return p
        except Exception as e:
            print(f"sqlite_vec.loadable_path() failed: {e}")
    except ImportError:
        print("sqlite_vec module not available")
    except Exception as e:
        print(f"Error importing sqlite_vec: {e}")
    
    # try common venv site-packages location
    guess = Path(sys.prefix) / "lib64" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "sqlite_vec" / "vec0.so"
    if guess.exists():
        print(f"Found vec0.so in venv site-packages: {guess}")
        return str(guess)
    
    # try user local installation
    local_guess = Path.home() / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "sqlite_vec" / "vec0.so"
    if local_guess.exists():
        print(f"Found vec0.so in user local installation: {local_guess}")
        return str(local_guess)
    
    # fallback: search in prefix
    try:
        for p in Path(sys.prefix).rglob("vec0.so"):
            print(f"Found vec0.so via prefix search: {p}")
            return str(p)
    except Exception as e:
        print(f"Error searching for vec0.so in prefix: {e}")
    
    print("vec0.so not found in any standard location")
    return None
def debug_sqlite_vec_module():
    """Debug sqlite_vec module to understand why it might not be working"""
    print("\n=== Debugging sqlite_vec module ===")
    try:
        import sqlite_vec
        print(f"sqlite_vec module loaded successfully: {sqlite_vec}")
        print(f"sqlite_vec module path: {sqlite_vec.__file__}")
        
        # Check available attributes
        print("Available attributes in sqlite_vec:")
        for attr in dir(sqlite_vec):
            if not attr.startswith('_'):
                print(f"  - {attr}")
        
        # Check if sqlite3 attribute exists
        if hasattr(sqlite_vec, 'sqlite3'):
            print(f"sqlite_vec.sqlite3: {sqlite_vec.sqlite3}")
            if hasattr(sqlite_vec.sqlite3, 'connect'):
                print("sqlite_vec.sqlite3.connect is available")
            else:
                print("sqlite_vec.sqlite3.connect is NOT available")
        else:
            print("sqlite_vec.sqlite3 attribute does NOT exist")
            
        # Check load function
        if hasattr(sqlite_vec, 'load'):
            print("sqlite_vec.load function is available")
        else:
            print("sqlite_vec.load function is NOT available")
            
    except ImportError as e:
        print(f"Cannot import sqlite_vec: {e}")
        print("Please ensure sqlite_vec is properly installed")
    except Exception as e:
        print(f"Error debugging sqlite_vec: {e}")
        traceback.print_exc()
def try_sqlite_vec_wrapper(vec_so):
    """Try sqlite_vec.sqlite3 wrapper - check if extension loading is supported"""
    print("\n=== Trying sqlite_vec.sqlite3 wrapper ===")
    try:
        import sqlite_vec
        print(f"sqlite_vec module imported: {sqlite_vec}")
        
        # Check if the wrapper is available
        if not hasattr(sqlite_vec, "sqlite3"):
            print("ERROR: sqlite_vec module missing 'sqlite3' attribute")
            debug_sqlite_vec_module()
            return None
            
        if not hasattr(sqlite_vec.sqlite3, "connect"):
            print("ERROR: sqlite_vec.sqlite3 missing 'connect' method")
            debug_sqlite_vec_module()
            return None
            
        if not hasattr(sqlite_vec, "load"):
            print("ERROR: sqlite_vec module missing 'load' function")
            debug_sqlite_vec_module()
            return None
            
        print("sqlite_vec.sqlite3 wrapper is available, attempting connection...")
        conn = sqlite_vec.sqlite3.connect(str(DB))
        print(f"Connection established: {conn}")
        
        # Try to load the extension
        print("Attempting to load vec0 extension...")
        sqlite_vec.load(conn)
        print("Extension loaded successfully!")
        
        # Verify it works
        version = conn.execute("select vec_version()").fetchone()[0]
        print(f"vec_version: {version}")
        
        return conn
        
    except ImportError:
        print("ERROR: sqlite_vec module not available")
        debug_sqlite_vec_module()
    except Exception as e:
        print(f"ERROR: sqlite_vec wrapper failed: {e}")
        debug_sqlite_vec_module()
        traceback.print_exc()
    return None
def try_stdlib_sqlite(vec_so):
    try:
        import sqlite3
        print("Trying stdlib sqlite3 with extension APIs...")
        conn = sqlite3.connect(str(DB))
        if not (hasattr(conn, "enable_load_extension") and hasattr(conn, "load_extension")):
            print("stdlib sqlite3 lacks extension APIs")
            return None
        
        try:
            import sqlite_vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            print("vec_version:", conn.execute("select vec_version()").fetchone()[0])
            return conn
        except ImportError:
            print("sqlite_vec module not available for stdlib method")
            return None
    except Exception as e:
        print(f"stdlib sqlite3 method failed: {e}")
        traceback.print_exc()
    return None
def try_apsw(vec_so):
    try:
        import apsw
        print("Trying APSW loadextension...")
        conn = apsw.Connection(str(DB))
        if not vec_so or not os.path.exists(vec_so):
            raise RuntimeError("vec0 .so not found for APSW")
        # APSW requires explicit enabling of extension loading on some builds
        try:
            conn.enable_load_extension(True)
        except Exception:
            # older/newer APSW builds may not expose this; ignore and try loadextension
            pass
        conn.loadextension(vec_so)
        try:
            cur = conn.cursor()
            cur.execute("select vec_version()")
            version = cur.fetchone()[0]
            print("vec_version:", version)
            return conn
        finally:
            try:
                conn.enable_load_extension(False)
            except Exception:
                pass
    except ImportError:
        print("APSW module not available")
    except Exception as e:
        print(f"APSW method failed: {e}")
        traceback.print_exc()
    return None
def try_ctypes(vec_so):
    try:
        import ctypes
        import ctypes.util
        print("Trying ctypes loader against local libsqlite3...")
        libsqlite = None
        # prefer user-local build
        local_lib = Path.home() / ".local" / "lib"
        for candidate in [local_lib / "libsqlite3.so.3.51.1", local_lib / "libsqlite3.so", local_lib / "libsqlite3.so.0"]:
            if candidate.exists():
                libsqlite = str(candidate)
                break
        if libsqlite is None:
            # fallback to system
            libsqlite = ctypes.util.find_library("sqlite3")
        if not libsqlite:
            print("libsqlite3 not found for ctypes loader")
            return None
        lib = ctypes.CDLL(libsqlite)
        # minimal required functions
        lib.sqlite3_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        lib.sqlite3_open.restype = ctypes.c_int
        lib.sqlite3_enable_load_extension.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.sqlite3_enable_load_extension.restype = ctypes.c_int
        lib.sqlite3_load_extension.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p)]
        lib.sqlite3_load_extension.restype = ctypes.c_int
        lib.sqlite3_errmsg.argtypes = [ctypes.c_void_p]
        lib.sqlite3_errmsg.restype = ctypes.c_char_p
        lib.sqlite3_close.argtypes = [ctypes.c_void_p]
        lib.sqlite3_close.restype = ctypes.c_int
        
        db = ctypes.c_void_p()
        rc = lib.sqlite3_open(str(DB).encode(), ctypes.byref(db))
        if rc != 0:
            print("sqlite3_open failed", rc); return None
        try:
            rc = lib.sqlite3_enable_load_extension(db, 1)
            errptr = ctypes.c_char_p()
            rc = lib.sqlite3_load_extension(db, vec_so.encode(), None, ctypes.byref(errptr))
            if rc != 0:
                msg = errptr.value.decode() if errptr and errptr.value else lib.sqlite3_errmsg(db).decode()
                print("load_extension failed:", rc, msg)
                return None
            print("ctypes: extension loaded OK")
            return True
        finally:
            lib.sqlite3_close(db)
    except Exception:
        traceback.print_exc()
    return None
def create_and_query(conn_obj):
    """Given a connection-like object, attempt to create virtual table and query.
    Supports sqlite3.Connection and APSW Connection objects.
    """
    try:
        # normalize to cursor-exec interface
        if hasattr(conn_obj, 'cursor'):
            cur = conn_obj.cursor()
            exec_fn = lambda q, params=(): cur.execute(q) if not params else cur.execute(q, params)
        else:
            # APSW Connection returns a cursor() with execute
            cur = conn_obj.cursor()
            exec_fn = lambda q, params=(): cur.execute(q)
        
        exec_fn("CREATE TABLE IF NOT EXISTS embeddings(id INTEGER PRIMARY KEY, embedding BLOB)")
        # create a simple vec0 virtual table; try multiple syntaxes to support
        # different vec0 builds (some report 'Unknown table option: id').
        candidates = [
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(id UNINDEXED, embedding FLOAT[3] distance_metric=L2)",
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(embedding FLOAT[3] distance_metric=L2)",
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(id INTEGER, embedding FLOAT[3] distance_metric=L2)",
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(rowid UNINDEXED, embedding FLOAT[3] distance_metric=L2)",
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(embedding BLOB)",
        ]
        exec_fn("DROP TABLE IF EXISTS vec_embeddings")
        last_err = None
        for sql in candidates:
            try:
                exec_fn(sql)
                print("Virtual table created OK with:", sql)
                return True
            except Exception as e:
                print("Candidate failed:", sql, "->", e)
                last_err = e
                # try next
                continue
        print("All CREATE VIRTUAL TABLE candidates failed; last error:", last_err)
        return False
    except Exception:
        traceback.print_exc()
    return False
def main():
    print("=== SQLite vec0 Test Script ===")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    vec_so = find_vec_so()
    print(f"Found vec0.so: {vec_so}")
    
    if not vec_so:
        print("ERROR: vec0.so not found. Please ensure it's installed and accessible.")
        print("You can set VEC0_SO environment variable to point to the library.")
        return
    
    if DB.exists():
        DB.unlink()
    
    print("\n=== Testing Registration Strategies ===")
    print("PRIORITY: APSW method")
    
    # Try registration strategies - PRIORITIZE APSW method
    results = {}
    
    # FIRST PRIORITY: APSW method
    conn = try_apsw(vec_so)
    if conn:
        ok = create_and_query(conn)
        results["APSW"] = ok
        print(f"SUCCESS: APSW method worked! Result: {ok}")
        if ok:
            print("\n=== TEST PASSED ===")
            print("APSW method is working correctly!")
        return
    else:
        results["APSW"] = False
        print("APSW method failed, trying fallback methods...")
    
    # SECOND PRIORITY: sqlite_vec wrapper (if extension loading is supported)
    print("\n=== Trying sqlite_vec.sqlite3 wrapper ===")
    conn = try_sqlite_vec_wrapper(vec_so)
    if conn:
        ok = create_and_query(conn)
        results["sqlite_vec wrapper"] = ok
        print(f"SUCCESS: sqlite_vec wrapper worked! Result: {ok}")
        if ok:
            print("\n=== TEST PASSED ===")
            print("sqlite_vec.sqlite3 wrapper is working correctly!")
        return
    else:
        results["sqlite_vec wrapper"] = False
        print("sqlite_vec wrapper failed or not supported, trying other fallback methods...")
    
    # THIRD PRIORITY: stdlib sqlite3
    conn = try_stdlib_sqlite(vec_so)
    if conn:
        ok = create_and_query(conn)
        results["stdlib sqlite3"] = ok
        print(f"Result with stdlib sqlite3: {ok}")
        return
    else:
        results["stdlib sqlite3"] = False
    
    # FOURTH PRIORITY: ctypes
    ct = try_ctypes(vec_so)
    results["ctypes"] = ct is not None
    print(f"ctypes result: {ct}")
    if ct:
        print("ctypes loaded extension; please use sqlite client or APSW to exercise virtual table creation")
        return
    
    print("\n=== Summary ===")
    print("Registration method results:")
    for method, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {method}: {status}")
    
    if not any(results.values()):
        print("\nAll registration strategies failed.")
        print("Recommendations:")
        print("  1. Ensure vec0.so is properly installed and accessible")
        print("  2. Check that APSW is installed (recommended method)")
        print("  3. Verify Python environment and module paths")
        print("  4. Try setting VEC0_SO environment variable explicitly")
        print(f"  5. Current vec0.so path: {vec_so}")
        print("  6. Run with debug output to understand the issue")

if __name__ == '__main__':
    main()
