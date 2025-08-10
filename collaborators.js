class CollaboratorManager {
  constructor() {
    this.collaborators = {};
    this.loadCollaborators();
  }

  async loadCollaborators() {
    try {
      const response = await fetch('./collaborators.json');
      const data = await response.json();
      this.collaborators = data.collaborators;
      this.updateCollaboratorLinks();
    } catch (error) {
      console.error('Failed to load collaborators:', error);
    }
  }

  getCollaborator(id) {
    return this.collaborators[id];
  }

  createCollaboratorLink(id) {
    const collaborator = this.getCollaborator(id);
    if (!collaborator) return id;
    
    return `<a href="${collaborator.url}">${collaborator.name}</a>`;
  }

  updateCollaboratorLinks() {
    // Update all elements with data-collaborator attribute
    document.querySelectorAll('[data-collaborator]').forEach(element => {
      const collaboratorId = element.getAttribute('data-collaborator');
      element.innerHTML = this.createCollaboratorLink(collaboratorId);
    });
  }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  new CollaboratorManager();
});
