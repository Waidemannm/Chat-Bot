w = np.array ([113,88,58,65,71,46,36,33,37,40,24,21,20,15,20])

def media_ponderada(vetor, pesos):
  return sum(vetor*pesos)/sum(pesos)

def media_ponderada2(vetor, pesos):
  media = 0
  for i in range(len(vetor)):
    media += vetor[i]*pesos[i]
  final = media / sum(pesos)
    
  return final

print(f'A média ponderada é: {media_ponderada(vetor,w):.2f}')
print(f'A média ponderada é: {media_ponderada2(vetor,w):.2f}') #segunda função