import timm

model = timm.create_model("hf_hub:cm93/resnet50-eurosat", pretrained=True)

print(model)
