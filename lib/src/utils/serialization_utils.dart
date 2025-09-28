class ModuleSerializationError extends StateError {
  ModuleSerializationError.unbuildLayer(String layer) : super("Layer $layer wasn't build and cannot be serialized");

  ModuleSerializationError.unbuildModel(String model)
    : super(
        "Model $model wasn't build and cannot be serialized. Please provide inputShape, call model on a data or manually invoke build method.",
      );
}
