def link_image(matched_image_names):
    import re

    # Pattern regex pour le préfixe à supprimer
    prefix_pattern = re.compile(r'./data1_image/[a-zA-Z_-]+/')

    # Pattern regex pour capturer la séquence de chiffres
    number_pattern = re.compile(r'(\d+)')

    # Liste pour stocker les chiffres extraits
    numbers = []

    # Boucle sur tous les liens
    for path in matched_image_names:
        # Utiliser la regex pour supprimer le préfixe spécifié
        path_without_prefix = re.sub(prefix_pattern, '', path)
        
        # Utiliser la regex pour extraire la séquence de chiffres
        match = re.search(number_pattern, path_without_prefix)
        
        if match:
            # Récupérer le nombre trouvé dans le groupe correspondant
            number = int(match.group(1))
            numbers.append(number)
        else:
            # Ajouter une valeur par défaut ou signaler une absence de nombre
            numbers.append(None)

    return numbers

