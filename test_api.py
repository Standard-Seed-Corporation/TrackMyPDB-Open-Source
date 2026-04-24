import requests
import json

# Test GraphQL query - corrected version
query = """
query ($pdbId: String!) {
  entry(entry_id: $pdbId) {
    polymer_entities {
      rcsb_polymer_entity_container_identifiers {
        reference_sequence_identifiers {
          database_accession
          database_name
        }
        uniprot_ids
      }
      rcsb_polymer_entity {
        pdbx_description
      }
    }
  }
}
"""

variables = {"pdbId": "3EQM"}

response = requests.post(
    "https://data.rcsb.org/graphql",
    json={"query": query, "variables": variables},
    headers={"Content-Type": "application/json"},
    timeout=15
)

print("Status Code:", response.status_code)
print("\nResponse:")
print(json.dumps(response.json(), indent=2))
