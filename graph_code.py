model.load_state_dict(torch.load('model_state_dict.pth'))
model.eval()

hidden = (torch.zeros(num_layers, 1, hidden_size),
               torch.zeros(num_layers, 1, hidden_size))

predictions = []
for i in tqdm(range(len(X_Train))):
    out, hidden = model(XTrain_tensor[i].unsqueeze(1), hidden)
    predictions.append(out.item())

plt.plot(Y_Train, label='Actual', linewidth=0.75)
plt.plot(predictions, label='Predicted', linewidth=0.75)
plt.ylabel('Price in USD')
plt.xlabel('Samples')
plt.legend()
plt.show()
