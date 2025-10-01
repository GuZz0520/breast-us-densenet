from .densenet import DenseNet121GAP

def build_model(backbone="densenet121", num_classes=3, pretrained=True,
                head_dropout=0.2, attention="none"):
    if backbone == "densenet121":
        return DenseNet121GAP(num_classes=num_classes,
                              dropout=head_dropout,
                              pretrained=pretrained,
                              attention=attention)
    else:
        raise NotImplementedError(f"backbone {backbone} not implemented")
