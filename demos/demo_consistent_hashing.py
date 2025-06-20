# demos/demo_consistent_hashing.py
import sys
import os
import time
import random
import hashlib # Para generar hashes de ejemplo

# Añadir la ruta raíz del proyecto al sys.path para importaciones
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.consistent_hashing.consistent_hasher import ConsistentHasher
from src.common.utils import log_info, log_warning

def run_consistent_hashing_demo():
    """
    Demostración del anillo de Hashing Consistente:
    distribución de claves y rebalanceo al añadir/remover nodos.
    """
    log_info("--- Iniciando demostración del Hashing Consistente ---")

    hasher = ConsistentHasher(num_replicas=100) # Más réplicas para una distribución más suave
    log_info(f"ConsistentHasher inicializado con {hasher.num_replicas} réplicas virtuales por nodo.")

    # 1. Añadir algunos nodos
    log_info("\n--- Añadiendo nodos al anillo ---")
    nodes = ["node_A", "node_B", "node_C"]
    for node_id in nodes:
        hasher.add_node(node_id)
        log_info(f"Nodo añadido: {node_id}")
    hasher.print_ring_distribution()

    # 2. Distribuir algunas claves
    log_info("\n--- Distribuyendo claves a los nodos actuales ---")
    keys_to_distribute = [f"vector_key_{i}" for i in range(15)] # 15 claves de ejemplo

    initial_assignments = {}
    for key in keys_to_distribute:
        assigned_node = hasher.get_node(key)
        initial_assignments[key] = assigned_node
        log_info(f"Clave '{key}' asignada a '{assigned_node}'")

    log_info("\nAsignaciones iniciales de claves:")
    for key, node in initial_assignments.items():
        log_info(f"  '{key}': '{node}'")

    # 3. Añadir un nuevo nodo y observar el rebalanceo
    log_info("\n--- Añadiendo 'node_D' y observando rebalanceo ---")
    hasher.add_node("node_D")
    log_info("Nodo añadido: node_D")
    hasher.print_ring_distribution()

    log_info("\nRevisando asignaciones después de añadir 'node_D':")
    rebalanced_assignments = {}
    changed_assignments = 0
    for key in keys_to_distribute:
        assigned_node = hasher.get_node(key)
        rebalanced_assignments[key] = assigned_node
        if assigned_node != initial_assignments[key]:
            log_info(f"  ¡Cambio! Clave '{key}' de '{initial_assignments[key]}' a '{assigned_node}'")
            changed_assignments += 1
        else:
            log_info(f"  Clave '{key}' permanece en '{assigned_node}'")
            
    log_info(f"Total de claves reasignadas: {changed_assignments} (idealmente bajo, ~1/N para Hashing Consistente)")

    # 4. Remover un nodo y observar el rebalanceo
    log_info("\n--- Removiendo 'node_B' y observando rebalanceo ---")
    hasher.remove_node("node_B")
    log_info("Nodo removido: node_B")
    hasher.print_ring_distribution()

    log_info("\nRevisando asignaciones después de remover 'node_B':")
    final_assignments = {}
    changed_assignments_after_remove = 0
    for key in keys_to_distribute:
        assigned_node = hasher.get_node(key)
        final_assignments[key] = assigned_node
        if assigned_node != rebalanced_assignments[key]:
            log_info(f"  ¡Cambio! Clave '{key}' de '{rebalanced_assignments[key]}' a '{assigned_node}'")
            changed_assignments_after_remove += 1
        else:
            log_info(f"  Clave '{key}' permanece en '{assigned_node}'")
            
    log_info(f"Total de claves reasignadas después de remover: {changed_assignments_after_remove}")


    log_info("\n--- Demostración del Hashing Consistente completada ---")

if __name__ == "__main__":
    run_consistent_hashing_demo()