from torch import nn
from torchvision.ops import Permute


class BasicLayer(nn.Module):
    def __init__(
            self, 
            channels, 
            inv_bottleneck_factor=4,
            kernel_size=5
    ):
        super().__init__()

        self.main_branch = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=kernel_size//2
            ),
            Permute([0, 2, 3, 1]),
            nn.LayerNorm(normalized_shape=channels),
            nn.Linear(in_features=channels, out_features=channels*inv_bottleneck_factor),
            nn.GELU(),
            nn.Linear(in_features=channels*inv_bottleneck_factor, out_features=channels),
            Permute([0, 3, 1, 2])
        )
    
    def forward(self, x):
        skip = x
        x = skip + self.main_branch(x)

        return x


class ExampleModel(nn.Module):
    def __init__(
            self, 
            feature_size, 
            num_stages,
            num_classes,
            stem_scale=2
    ):
        super().__init__()
        
        self.hidden_sizes = feature_size
        
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=3, 
                out_channels=feature_size, 
                kernel_size=stem_scale, 
                stride=stem_scale
            ),
            Permute([0, 2, 3, 1]),
            nn.LayerNorm(normalized_shape=feature_size),
            Permute([0, 3, 1, 2])
        )

        self.stages = nn.ModuleList()
        stage_feature_size = feature_size
        for stage in range(num_stages):
            if stage > 0:
                self.stages.append(
                    nn.Sequential(
                        Permute([0, 2, 3, 1]),
                        nn.LayerNorm(normalized_shape=stage_feature_size),
                        Permute([0, 3, 1, 2]),
                        nn.Conv2d(
                            in_channels=stage_feature_size, 
                            out_channels=stage_feature_size*2, 
                            kernel_size=2, 
                            stride=2
                        )
                    )
                )
                stage_feature_size *= 2

            for i in range(2):
                self.stages.append(
                    BasicLayer(
                        channels=stage_feature_size,
                    )
                )
            
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=stage_feature_size),
            nn.Linear(
                in_features=stage_feature_size, 
                out_features=num_classes, 
                bias=False
            )
        )

    def forward(self, x):
        # Stem
        x = self.stem(x)

        # Stages
        for stage in self.stages:
            x = stage(x)
        
        # Globale average
        x = x.mean(dim=[-2, -1])

        # Classification
        x = self.classifier(x)

        return x