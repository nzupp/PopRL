"""
Integration of stdpopsim into PopRL, allowing any specified demographic model
to be turned into a reinforcement learning env
"""
import stdpopsim

# Create massive tables when pretty printing demographies, pruned for visual improvement
SKIP_TYPES = {"MigrationRateChange"}
SKIP_FIELDS = {"demography", "population_id"}

def get_model(species_id, model_id):
    """Pulls the demographic model from stdpopsim, using species and model ID"""
    try:
        sp = stdpopsim.get_species(species_id)
    
    except ValueError:
        print(f"Species '{species_id}' not found.")
        return None, None
    
    try:
        m = sp.get_demographic_model(model_id)
    
    except ValueError:
        print(f"Model '{model_id}' not found for species '{species_id}'.")
        return None, None
    
    return m.model, m.mutation_rate

def get_model_info(m):
    """Get specific parameters of a given demographic model from stdpopsim"""
    demog = m.model
    events = [
        parse_event(e) for e in demog.events
        if type(e).__name__ not in SKIP_TYPES
    ]
    
    return {
        "id": m.id,
        "generation_time": m.generation_time,
        "mutation_rate": m.mutation_rate,
        "populations": [
            {"name": p.name, "initial_size": p.initial_size, "growth_rate": p.growth_rate}
            for p in demog.populations
        ],
        "migration_matrix": demog.migration_matrix.tolist(),
        "events": events,
    }

def avail_stdpopsim(species=None):
    """Pretty print all availabe stdpopsim models, able to query by species"""
    if species is None:
        for s in stdpopsim.all_species():
            if not s.demographic_models:
                continue
            print_species(s)
        return
    
    try:
        sp = stdpopsim.get_species(species)
    
    except ValueError:
        print(f"Species '{species}' not found.")
        return
    
    if not sp.demographic_models:
        print(f"No models for {species}.")
        return
    
    print_species(sp)

def parse_event(e):
    """Parse demographic events in stdpopsim histories"""
    d = {k: v for k, v in vars(e).items() if k not in SKIP_FIELDS}
    d["type"] = type(e).__name__
    return d

def print_species(s):
    """Pretty print helping function"""
    print(f"\n{'='*60}")
    print(f"SPECIES: {s.id} - {s.name}")
    print(f"{'='*60}")
    
    for m in s.demographic_models:
        info = get_model_info(m)
        print(f"\n  --- Model: {info['id']} ---")
        print(f"  generation_time : {info['generation_time']}")
        print(f"  mutation_rate   : {info['mutation_rate']}")
        print("  populations     :")
        
        for p in info['populations']:
            print(f"    {p}")
        
        print(f"  migration_matrix: {info['migration_matrix']}")
        print("  events          :")
        
        for e in info['events']:
            print(f"    {e}")
            

