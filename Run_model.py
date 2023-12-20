# Importa le classi del modello e degli agenti
from Model.AdaptationModel import AdaptationModel

# Crea un'istanza del modello
adaptation_model = AdaptationModel(
    # Qui puoi impostare i parametri iniziali se necessario
)

# Esegui la simulazione per un certo numero di passi
for i in range(100):  # Ad esempio, 100 passi
    adaptation_model.step()

# Al termine della simulazione, visualizza i risultati
adaptation_model.plot_model_domain_with_agents()